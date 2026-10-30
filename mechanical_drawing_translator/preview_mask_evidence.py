# mechanical_drawing_translator/preview_mask_evidence.py
"""Private native-object evidence; raster agreement cannot authorize erasure.

A connected short engineering stroke can survive every OCR/component test.
The product therefore removes matched native text objects in a private render
and checks the remaining geometry. Raster proposals stay research evidence.
"""
import re,unicodedata
from dataclasses import dataclass
from PySide6.QtCore import QThread,Signal
from PySide6.QtGui import QImage


# 后台结果带不可变文档、页和目标快照返回。
# 接收方可据此拒绝切页或新请求之后到达的旧结果。
# 该请求不携带人工审核授权，翻译来源与擦除证据分别处理。
# entries 转为 tuple，防止 UI 改写目标列表影响正在工作的线程。
@dataclass(frozen=True)
class MaskEvidenceRequest:
    document_id: str
    page_index: int
    entries: tuple


# 耗时 PDFium 私有渲染在后台执行，避免阻塞主界面。
# 取消是软中断：在边界检查并抑制结果投递，不强杀底层渲染。
# 源图复制为任务私有数据，避免 UI 替换缓冲区后证据失去对应。
# 任务仅调用原生对象去字证据，栅格多尺度研究路径不在此接线。
# 失败总是返回空 verified，而不是沿用旧项目的成功集合。
class MaskEvidenceTask(QThread):
    """Private PDFium work stays off the UI thread; cancellation suppresses delivery."""
    result_ready=Signal(object,object)
    def __init__(self,path,geometry,entries,width,parent=None,*,source_image=None,obstacles=(),expected_source_sha=None):
        super().__init__(parent)
        self.request=MaskEvidenceRequest(geometry.document_id,geometry.page_index,tuple(entries))
        self.path=path;self.geometry=geometry;self.width=width
        self.source_image=source_image.copy() if source_image is not None else None
        self.obstacles=tuple(obstacles);self.expected_source_sha=expected_source_sha
    def run(self):
        if self.isInterruptionRequested():return
        from .project_store import source_sha256
        try:
            # 当存在预期 SHA 或源图时，工作前记录实际源内容身份。
            # 文件不可读也必须形成失败身份，不能跳过后续不一致判断。
            # 文件名和路径相同不保证内容相同，外部编辑可能替换字节。
            # 不匹配时返回明确 rejection，不能渲染新文件配旧 OCR 坐标。
            before=source_sha256(self.path) if self.expected_source_sha is not None or self.source_image is not None else None
        except OSError:before='unavailable'
        if self.expected_source_sha is not None and before!=self.expected_source_sha:
            if self.source_image is None:evidence=(QImage(),frozenset())
            else:
                from .raster_pdf_evidence import PreviewMaskEvidence
                evidence=PreviewMaskEvidence(self.source_image.copy(),frozenset(),{},
                    {e.group_id:'source_sha_mismatch' for e in self.request.entries},())
            if not self.isInterruptionRequested():self.result_ready.emit(self.request,evidence)
            return
        # 当前唯一生产授权来源是原生文字对象在私有实例中停绘后的背景。
        # 源哈希工作后再检验，检测分析期间外部写入造成的竞态。
        # 发生改变撤销整包证据，原图副本与诊断保留供界面解释。
        # 取消后不 emit，避免旧任务结果覆盖当前页状态。
        evidence=native_mask_evidence(self.path,self.request.page_index,self.geometry,self.request.entries,self.width)
        if before is not None:
            try:changed=source_sha256(self.path)!=before
            except OSError:changed=True
            if changed:
                if self.source_image is None:evidence=(QImage(),frozenset())
                else:
                    from .raster_pdf_evidence import PreviewMaskEvidence
                    evidence=PreviewMaskEvidence(self.source_image.copy(),frozenset(),{},
                        {e.group_id:'source_changed_during_evidence' for e in self.request.entries},())
        if not self.isInterruptionRequested():self.result_ready.emit(self.request,evidence)


# 只在原生对象字面比对时做 NFKC、空白压缩和大小写归一。
# 不会改写持久原文，也不从中文倒推英文。
# 对象选择还需几何包含与完整源跨度，不能单靠归一化文本匹配。
def _text(value):return re.sub(r'\s+','',unicodedata.normalize('NFKC',value)).casefold()


def native_mask_evidence(path,page_index,geometry,entries,render_width):
    # 原生遮挡对工程数值和关键符号另设禁入门槛。
    # 共享框无法证明只停绘本译文对应字面，因此不接受。
    # OCR 来源即使高置信、人工批准或有组件蒙版也不进入 eligible。
    # 这里先缩小候选，布局边界还会确认完整区域与 SourceSpan 覆盖。
    # 空候选直接返回空证据，避免无必要打开 PDF。
    eligible=[e for e in entries if e.coordinate_precision!='shared_region_bounds'
              and all(r.source=='pdf_native' for r in e.regions)
              and not re.search(r'[0-9±Ø⌀°]',e.source_text)]
    if not eligible:return QImage(),frozenset()
    try:
        import pypdfium2 as pdfium
        from pypdfium2 import raw
        with pdfium.PdfDocument(str(path)) as pdf:
            page=pdf[page_index]
            try:
                # 仅支持原点标准、CropBox 等于 MediaBox、未旋转且尺寸匹配的页。
                # 这避免把 PDF 底部原点和 Qt 顶部原点的转换用于未知页面变换。
                # 不支持页直接保留英文，不能为扩大成功数放宽 transform 条件。
                # 渲染实例私有，停绘对象不会影响主界面已打开的 PDF。
                w,h=page.get_size();media=page.get_mediabox();crop=page.get_cropbox()
                if (page.get_rotation()!=0 or media!=crop or abs(media[0])+abs(media[1])>.01
                    or abs(w-geometry.width)>.1 or abs(h-geometry.height)>.1):return QImage(),frozenset()
                tp=page.get_textpage()
                try:
                    # 只枚举顶层原生文字对象，深层 Form 中的未知关联不猜测处理。
                    # 提取对象字面与 bounds 的证据必须来自同一 textpage。
                    # textpage 及时关闭，但页面和对象仍留在本次私有文档生命周期内。
                    # 本路径没有调用 PDF writer/save，原文件内容从未被改写。
                    objects=[(o,o.extract(),o.get_bounds()) for o in page.get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT],max_depth=1,textpage=tp)]
                finally:tp.close()
                selected={};verified=set()
                for e in eligible:
                    b=e.bounds
                    # 候选原生对象须完整落在目标框的窄容差内。
                    # PDF 原生 y 自底向上，用 h-顶/底映射到布局左上方向。
                    # 先拼接全部包含对象，再与源组字面精确归一化对照。
                    # 不完整、重复或夹杂其它对象导致不一致时不给 verified。
                    # 跨组共享对象用索引去重，只停绘一次。
                    contained=[(i,o,text,rect) for i,(o,text,rect) in enumerate(objects)
                               if text.strip() and rect[0]>=b.min_x-1 and rect[2]<=b.max_x+1
                               and h-rect[3]>=b.min_y-1 and h-rect[1]<=b.max_y+1]
                    if _text(''.join(t for _,_,t,_ in contained))!=_text(e.source_text):continue
                    verified.add(e.group_id)
                    for i,o,_,_ in contained:selected[i]=o
                if not selected:return QImage(),frozenset()
                # 每个选中对象必须成功停绘，否则整页不产生半可信背景。
                # 全部 matched 对象一起停绘后重新渲染，剩余短线仍应可见。
                # 这种独立背景让下游能检查蒙版内部，而不只看文字四角。
                # 它仍需其它源框碰撞复验，verified 不是最终 SAFE。
                for o in selected.values():
                    if not raw.FPDFPageObj_SetIsActive(o,False):return QImage(),frozenset()
                # Mutations exist only in this private PDFium instance. No PDF
                # writer/save API is called; original bytes and Qt page remain.
                # 渲染宽度至少保持一倍 PDF point 尺度，供源坐标空白检测。
                # 转换 RGB 只用于证据工作图，正式 PDF 导出仍保留原页对象。
                # QImage copy 脱离 Pillow 字节缓冲，位图关闭后结果仍有效。
                # finally 层次确保 bitmap、page 和文档按所有路径释放。
                bitmap=page.render(scale=max(1,render_width/w))
                try:
                    image=bitmap.to_pil().convert('RGB');data=image.tobytes()
                    result=QImage(data,image.width,image.height,image.width*3,QImage.Format.Format_RGB888).copy()
                finally:bitmap.close()
                return result,frozenset(verified)
            finally:page.close()
    # 底层 API 缺失、损坏 PDF 或任何解析异常都降级为空证据。
    # 此宽异常边界用于禁止误擦，不能把它写成背景已安全验证。
    # 无证据影响遮挡选项，双语中文的邻近布局仍可单独成立。
    except Exception:
        # Unavailable native API or malformed PDF evidence always preserves ink.
        return QImage(),frozenset()
