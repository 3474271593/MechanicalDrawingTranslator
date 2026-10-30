# mechanical_drawing_translator/recovery_store.py
"""Local atomic recovery copies; restore never writes the formal project.

The checksum detects partial/corrupt drafts, not hostile modification. Every
restore also passes the normal project identity/geometry/provenance validator.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .project_store import source_sha256,encode


# 输入是主线程脱离控件的快照
# 后台仅操作Python数据
# 源SHA在编码前再次读取
# 防止打开后源图已被替换
# 普通保存也复用此契约
# 各缓存按统一encode编码
# geometry逐页保留元数据
# PDF再记录旋转与页面盒
# 不把QPdfDocument传给后台
# 不改变正式项目文件
# 写草稿前还会再核验SHA
def project_payload_from_snapshot(snapshot):
    """Encode only detached Python data; this function never touches a widget.

    The source is rechecked here and again before atomic draft replacement.
    Both ordinary saves and background recovery use this same payload contract.
    """
    payload=dict(snapshot)
    if source_sha256(payload['source']['path'])!=payload['source']['sha256']:
        raise ValueError('原图纸打开后发生变化；请重新识别')
    for key in ('document','ocr','targets','translations','layouts','manual_layouts','machine_suggestions'):
        payload[key]=encode(snapshot[key])
    payload['geometry']=[encode(g) for g in snapshot['geometry']]
    if payload['source']['format']=='PDF':
        from pypdf import PdfReader
        reader=PdfReader(payload['source']['path'])
        payload['pdf_page_boxes']=[{'page_index':i,'rotate':int(p.get('/Rotate',0)),
            'cropbox':[float(v) for v in p.cropbox],'mediabox':[float(v) for v in p.mediabox]}
            for i,p in enumerate(reader.pages)]
    return payload


# 环境变量可隔离恢复目录
# 默认用用户LocalAppData
# 不能把草稿写入源代码包
# 源码启动和冻结包共享规则
# 不承诺额外加密或独立ACL
def default_recovery_directory():
    configured=os.environ.get('MDT_RECOVERY_DIR')
    if configured:return Path(configured)
    return Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')/'MechanicalDrawingTranslator'/'recovery'


# 按排序键生成稳定JSON
# 紧凑分隔避免格式差异
# UTF-8字节计算校验和
# 只检测损坏与不完整草稿
# 不提供抗恶意修改签名
def _checksum(payload):
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,allow_nan=False,
        sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


class RecoveryDraftStore:
    def __init__(self,directory=None):
        self.directory=Path(directory) if directory is not None else default_recovery_directory()

    # 路径哈希决定草稿文件名
    # Windows规范化大小写
    # resolve避免相对路径歧义
    # 不把原始文件名明文放目录名
    # 内容变化仍能找到旧草稿
    # 源内容SHA单独写入信封
    # 因此可报告source_conflict
    def path_for(self,source):
        # Path identity permits reporting a changed source instead of losing its
        # draft when the content hash changes. The envelope separately binds SHA.
        key=os.path.normcase(str(Path(source).resolve()))
        return self.directory/(hashlib.sha256(key.encode('utf-8')).hexdigest()+'.draft.json')

    # 先核验快照对应源字节
    # 目标不得等于源或正式项目
    # 草稿信封绑定源SHA
    # formal_path绑定正式文件位置
    # base_formal_sha记录打开时版本
    # payload_sha检测草稿内容损坏
    # saved_at使用UTC时区
    # 临时文件写完后fsync
    # replace只替换恢复副本
    # 不顺带更新正式审核文件
    # finally清理未提交临时文件
    # 返回实际草稿路径供追溯
    def write(self,payload,*,formal_path=None,base_formal_sha=None):
        source=Path(payload['source']['path']).resolve()
        if source_sha256(source)!=payload['source']['sha256']:
            raise ValueError('原图纸已改变，不能写入旧坐标恢复草稿')
        destination=self.path_for(source)
        formal=Path(formal_path).resolve() if formal_path else None
        if destination.resolve()==source or (formal and destination.resolve()==formal):
            raise ValueError('恢复草稿不能覆盖源文件或正式项目')
        envelope={'schema_version':1,'kind':'mdt_recovery_draft',
            'saved_at':datetime.now(timezone.utc).isoformat(),
            'source_sha256':payload['source']['sha256'],
            'formal_path':str(formal) if formal else None,'base_formal_sha256':base_formal_sha,
            'payload':payload,'payload_sha256':_checksum(payload)}
        self.directory.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix=destination.name+'.',suffix='.tmp',dir=self.directory)
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
                json.dump(envelope,stream,ensure_ascii=False,allow_nan=False)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,destination)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        return destination

    # 缺失草稿与无效草稿分开
    # 先验schema与kind
    # 再验payload自身校验和
    # 路径与信封源SHA必须一致
    # 实际源变化报告source_conflict
    # 正式文件变化报告formal_conflict
    # 正式文件被删除也算冲突
    # 没有formal却有base_sha属非法
    # 可用仅说明身份门禁通过
    # 恢复还需完整项目校验
    # 损坏JSON不向外抛解析异常
    # 不会自动合并较新人工审核
    def _inspect(self,source):
        path=self.path_for(source)
        if not path.is_file():return 'missing',None
        try:
            envelope=json.loads(path.read_text('utf-8'))
            if envelope.get('schema_version')!=1 or envelope.get('kind')!='mdt_recovery_draft':
                return 'invalid',None
            payload=envelope['payload']
            if envelope['payload_sha256']!=_checksum(payload):return 'invalid',None
            if (Path(payload['source']['path']).resolve()!=Path(source).resolve()
                or payload['source']['sha256']!=envelope['source_sha256']):return 'invalid',None
            if source_sha256(source)!=envelope['source_sha256']:return 'source_conflict',envelope
            formal=envelope.get('formal_path');expected=envelope.get('base_formal_sha256')
            if formal:
                actual=source_sha256(formal) if Path(formal).is_file() else None
                if actual!=expected:return 'formal_conflict',envelope
            elif expected is not None:return 'invalid',None
            return 'available',envelope
        except FileNotFoundError:return 'source_conflict',None
        except (OSError,ValueError,KeyError,TypeError):return 'invalid',None

    # 只暴露状态给界面
    # 不因查看状态删除草稿
    def status(self,source):
        return self._inspect(source)[0]

    # 只有available允许读取恢复
    # 冲突状态明确拒绝
    # 不提供强制跳过SHA接口
    def read(self,source):
        status,envelope=self._inspect(source)
        if status!='available':raise ValueError('恢复草稿不可恢复：'+status)
        return envelope

    # 只删除该源对应的草稿
    # 正式项目和源图均不触及
    # 无草稿时返回False
    # 用户选择由上层明确处理
    def discard(self,source):
        path=self.path_for(source)
        if not path.exists():return False
        path.unlink();return True
