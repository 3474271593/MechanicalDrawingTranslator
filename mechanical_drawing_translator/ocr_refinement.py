# mechanical_drawing_translator/ocr_refinement.py
"""Bounded image-backed OCR rescue; immutable originals and conservative adoption."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
from pathlib import Path
import re
from time import perf_counter
from typing import Sequence

from .models import TextBounds, TextRegion

REFINEMENT_VERSION = "ocr-3c-v1"


# 每条候选绑定识别方法
# image_sha指向实际ROI像素
# DPI记录候选渲染尺度
# 高置信不是自动采用许可
@dataclass(frozen=True)
class LocalCandidate:
    text: str
    confidence: float
    method: str
    image_sha256: str = ""
    dpi: int = 0


# 父区域身份始终保留
# 候选和采用结果分开
# decision解释接受或拒绝
# numeric_views保留双尺度读数
# 原始OCR随记录可回溯
# 耗时单列局部复查成本
@dataclass(frozen=True)
class RecoveryRecord:
    region_id: str
    trigger: str
    candidates: tuple[LocalCandidate, ...]
    adopted: LocalCandidate | None
    decision: str
    numeric_views: tuple[tuple[str, ...], ...]
    elapsed_seconds: float
    original_region: TextRegion | None = None


# selected供后续目标筛选
# records供复核和回滚
# parent映射连接新旧区域
# 原始OCR仍在OcrPage.regions
# 告警不隐瞒复查失败
@dataclass(frozen=True)
class RefinementReport:
    selected_regions: tuple[TextRegion, ...]
    records: tuple[RecoveryRecord, ...]
    parent_by_region_id: dict[str, str]
    elapsed_seconds: float
    warnings: tuple[str, ...] = ()


# 仅针对低置信稀疏OCR框
# 不修改原生PDF文字
# 阅读角度限近水平
# 宽长框中只识出少数字母
# 这只是复查触发条件
# 不代表已确认漏字
def _sparse(region: TextRegion) -> bool:
    width = region.bounds.max_x - region.bounds.min_x
    height = max(1, region.bounds.max_y - region.bounds.min_y)
    letters = re.findall(r"[A-Za-z]", region.source_text)
    return bool(region.source == "ocr" and region.confidence is not None
                and region.confidence < .8 and abs(region.rotation_degrees) <= 15
                and 0 < len(letters) <= 4 and width >= 40 and width/height >= 8)


# 严重稀疏框最多取两个
# 密集小字按几何聚类
# 同文档同页才可相邻
# 行高决定距离容差
# 连续三行才形成抽样簇
# 取首行与中间行复核
# 密集样本最多再取两个
# 每页总工作保持有界
# 密集高置信只复核不改字
def refinement_triggers(regions: Sequence[TextRegion]) -> tuple[tuple[TextRegion, str], ...]:
    severe = [(r, "sparse_low_confidence") for r in regions if _sparse(r)][:2]
    small = sorted([r for r in regions if r.source == "ocr" and abs(r.rotation_degrees) <= 15
                    and r.bounds.max_y-r.bounds.min_y <= 7
                    and len(re.findall(r"[A-Za-z]+",r.source_text)) >= 3],
                   key=lambda r:(r.page_index,r.bounds.min_y,r.bounds.min_x))
    clusters: list[list[TextRegion]] = []
    for r in small:
        if clusters:
            previous=clusters[-1][-1]
            height=max(1,r.bounds.max_y-r.bounds.min_y,previous.bounds.max_y-previous.bounds.min_y)
            if (r.document_id == previous.document_id and r.page_index == previous.page_index
                    and 0 <= r.bounds.min_y-previous.bounds.min_y <= height*1.4
                    and abs(r.bounds.min_x-previous.bounds.min_x) <= height):
                clusters[-1].append(r)
                continue
        clusters.append([r])
    sampled=[]
    for cluster in clusters:
        if len(cluster)>=3:
            sampled.extend((cluster[0],cluster[len(cluster)//2]))
    return tuple(severe+[(r,"small_dense_line_review") for r in sampled[:2]])


# 只为候选共识拆词
# 数字假设另交像素核验
# A/B标注不能忽略
# A1/A2也必须参与比较
# 不把字母O改成零
# 不把I改成数字一
# 原始字面始终另存
def _words(text: str) -> tuple[str, ...]:
    # Ignore only decimal/numeric hypotheses. Callouts A/B and IDs A1/A2
    # must participate in consensus; never map O->0 or I->1.
    tokens=[]
    for word in text.split():
        token=word.strip("*[](){} ,:;").upper()
        numeric=(re.fullmatch(r"[+\-.0-9OI]+",token)
                 and (any(c.isdigit() for c in token) or token.startswith(".")))
        if token and not numeric:
            tokens.append(token.strip(".,;:"))
    return tuple(tokens)


# 保留小数点与正负号
# 不把单词内数字随意抽离
# 返回序列供顺序一致核验
def _numbers(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![\w.])[+-]?(?:\d*\.\d+|\d+)(?![\w.])",text))


# 括号星号也是工程信息
# 候选采用需比较这些标记
# 不能只比较英文字母正文
def _marks(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\[\](){}*\"'%]",text))


# 只允许稀疏低置信自动救回
# 其他触发一律review_only
# 任一解释含危险工程符号即弃权
# 不能偏选漏掉符号的候选
# 两尺度direct正文先达成共识
# 星号和括号也必须相同
# 至少五个词支撑正文恢复
# 双尺度数字像素必须一致
# 原否定/条件词不得丢失
# 原字母数字ID必须保留
# 已有数字必须有像素支持
# 最终候选再检数字完整序列
# 疑似OI数字混淆继续拒绝
# 合格候选中取最高置信
# 返回原因用于UI审计
# 没有合格项安全保留原OCR
def choose_recovery(region: TextRegion, candidates: Sequence[LocalCandidate],
                    numeric_views: Sequence[tuple[str, ...]]) -> tuple[LocalCandidate | None,str]:
    if not _sparse(region):
        return None,"high_confidence_review_only"
    # A competing symbol-bearing interpretation is a reason to abstain,
    # not a reason to prefer the candidate which silently omitted it.
    if any(re.search(r"[±°Øø⌀∅/\\=<>]",text) for text in
           [region.source_text,*(c.text for c in candidates)]):
        return None,"engineering_symbol_requires_review"
    direct=[c for c in candidates if c.method.startswith("direct-") and c.confidence>=.85]
    if (len(direct)<2 or _words(direct[0].text)!=_words(direct[1].text)
            or _marks(direct[0].text)!=_marks(direct[1].text)):
        return None,"body_candidate_conflict"
    words=_words(direct[0].text)
    if len(words)<5:
        return None,"insufficient_body_evidence"
    if len(numeric_views)<2 or not numeric_views[0] or numeric_views[0]!=numeric_views[1]:
        return None,"numeric_pixel_evidence_missing_or_conflicting"
    old_conditions=set(re.findall(r"\b(?:NO|NOT|ONLY|BEFORE|AFTER|UNLESS|WITHOUT)\b",region.source_text.upper()))
    old_numbers=_numbers(region.source_text)
    old_ids={t for t in _words(region.source_text)
             if any(c.isalpha() for c in t) and any(c.isdigit() for c in t)}
    if (not (old_conditions|old_ids).issubset(words)
            or any(n not in numeric_views[0] for n in old_numbers)
            or (_marks(region.source_text) and _marks(region.source_text)!=_marks(direct[0].text))):
        return None,"protected_original_condition_or_numeric_conflict"
    eligible=[]
    for c in candidates:
        if (_words(c.text)==words and c.confidence>=.9
                and _marks(c.text)==_marks(direct[0].text)
                and _numbers(c.text)==numeric_views[0]
                and not re.search(r"[±°Øø⌀∅/\\=<>]",c.text)
                and not re.search(r"[.\[][A-Za-z0-9.]*[OI][A-Za-z0-9.]*\d",c.text)):
            eligible.append(c)
    if not eligible:
        return None,"numeric_or_body_candidate_conflict"
    return max(eligible,key=lambda c:c.confidence),"body_and_numeric_image_consensus"


# 先验证记录属于当前页
# 同一父区域不能重复采用
# 未采用者复用原冻结对象
# 采用者生成新的确定性ID
# ID绑定版本/文档/页/字面
# 新文本不复用旧翻译身份
# 状态标明可回滚局部识别
# parent连接新ID到旧ID
# records补回完整原区域
# 不能伪造字符级几何
# 整体边界仍沿用原框
def apply_recoveries(regions: Sequence[TextRegion], records: Sequence[RecoveryRecord]) -> RefinementReport:
    by_id={r.id:r for r in regions}
    if len({r.region_id for r in records})!=len(records) or any(r.region_id not in by_id for r in records):
        raise ValueError("Recovery region does not belong to this page or is duplicated")
    adopted={r.region_id:r.adopted for r in records if r.adopted is not None}
    selected=[]
    parents={}
    for original in regions:
        candidate=adopted.get(original.id)
        if candidate is None:
            selected.append(original)
            parents[original.id]=original.id
            continue
        identity=hashlib.sha256((REFINEMENT_VERSION+'|'+original.document_id+'|'
                                +str(original.page_index)+'|'+original.id+'|'+candidate.text).encode()).hexdigest()
        new=replace(original,id="roi-"+identity,source_text=candidate.text,
                    confidence=candidate.confidence,status="局部复识别（可回滚）")
        selected.append(new)
        parents[new.id]=original.id
    traced=tuple(replace(r,original_region=by_id[r.region_id]) for r in records)
    return RefinementReport(tuple(selected),traced,parents,
                            sum(r.elapsed_seconds for r in records))


# 以PNG字节固定候选图像身份
# 哈希仅是证据关联
# 不是正文专业正确性证明
def _image_hash(image) -> str:
    stream=BytesIO(); image.save(stream,format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


# 只从图像组件读取数字
# 不从英文单词猜小数
# 先选最靠近ROI中心的文字带
# 水平投影分离组件
# 组件数超过24直接弃权
# 控制病态ROI额外推理成本
# 每片补白避免边缘裁字
# 仅接受单结果高置信数值
# 允许外围方括号与空白
# 返回顺序与图像从左到右一致
def numeric_pixel_view(image,engine) -> tuple[str, ...]:
    """Read separated image components; never infer numeric characters from words."""
    import numpy as np
    from PIL import ImageOps
    ink=np.asarray(image.convert("L"))<160
    yy=np.flatnonzero(ink.sum(axis=1)>image.width*.03)
    if not len(yy): return ()
    bands=np.split(yy,np.flatnonzero(np.diff(yy)>1)+1)
    usable=[b for b in bands if len(b)>3]
    if not usable: return ()
    band=min(usable,key=lambda b:abs((b[0]+b[-1])/2-image.height/2))
    line=image.crop((0,int(band[0]),image.width,int(band[-1]+1)))
    xx=np.flatnonzero((np.asarray(line.convert("L"))<180).any(axis=0))
    if not len(xx): return ()
    parts=np.split(xx,np.flatnonzero(np.diff(xx)>line.height*.65)+1)
    if len(parts)>24: return ()  # Bound the total rec-only work for pathological crops.
    numbers=[]
    for part in parts:
        patch=ImageOps.expand(line.crop((int(part[0]),0,int(part[-1]+1),line.height)),
                              border=4,fill="white")
        reads=engine.recognize_local(patch)
        if len(reads)==1:
            text=reads[0].text.strip()
            if reads[0].confidence>=.9 and re.fullmatch(r"[\[\]\s]*[+-]?(?:\d*\.\d+|\d+)[\[\]\s]*",text):
                numbers.extend(_numbers(text))
    return tuple(numbers)


# 没有触发项不打开额外文档
# 每个ROI增加一点边距
# 坐标裁剪在源页范围内
# 面积超过上限记录拒绝
# 216和360 DPI独立渲染
# crop按PDFium四边距离转换
# y0/y1需与页面高度换算
# 每张工作图单独记录SHA
# 证据图仅显式目录时写盘
# 普通生产不会默认留整图
# 先做关闭检测的局部识别
# 稀疏框再读双尺度数字
# 对比度候选只是额外证据
# 局部检测只取中心带
# 邻行碎片不能拼进正文
# 采用仍由choose_recovery统一决定
# 记录保留所有候选与耗时
# finally关闭页面和文档
def refine_pdf_page(path: Path, page_index: int, regions: Sequence[TextRegion],engine,
                    *, evidence_dir: Path | None = None) -> RefinementReport:
    import pypdfium2 as pdfium
    from PIL import ImageEnhance, ImageOps
    triggers=refinement_triggers(regions)
    if not triggers: return apply_recoveries(regions,())
    records=[]
    document=pdfium.PdfDocument(str(path))
    page=document[page_index]
    try:
        width,height=page.get_size()
        for region,trigger in triggers:
            started=perf_counter(); candidates=[]; images=[]
            b=region.bounds; pad=1.0
            x0,y0,x1,y1=max(0,b.min_x-pad),max(0,b.min_y-pad),min(width,b.max_x+pad),min(height,b.max_y+pad)
            if (x1-x0)*(y1-y0)>12000:
                records.append(RecoveryRecord(region.id,trigger,(),None,"roi_area_limit",(),0))
                continue
            for dpi in (216,360):
                bitmap=page.render(scale=dpi/72,crop=(x0,height-y1,width-x1,y0))
                image=bitmap.to_pil().convert("RGB"); bitmap.close(); images.append(image)
                image_hash=_image_hash(image)
                if evidence_dir is not None:
                    evidence_dir.mkdir(parents=True,exist_ok=True)
                    image.save(evidence_dir/f"{region.id}-{dpi}.png")
                reads=engine.recognize_local(image)
                candidates.extend(LocalCandidate(r.text,r.confidence,f"direct-{dpi}",image_hash,dpi) for r in reads)
            numeric_views=()
            if trigger=="sparse_low_confidence":
                numeric_views=tuple(numeric_pixel_view(im,engine) for im in images)
                enhanced=ImageEnhance.Contrast(ImageOps.autocontrast(images[0])).enhance(1.8)
                image_hash=_image_hash(enhanced)
                reads=engine.recognize_local(enhanced,detect=True)
                # Local detection may include neighbouring line fragments; only accept the central band.
                for r in reads:
                    ys=[p[1] for p in r.polygon]
                    if abs((min(ys)+max(ys))/2-enhanced.height/2)<=enhanced.height*.3:
                        candidates.append(LocalCandidate(r.text,r.confidence,"det-contrast-216",image_hash,216))
            adopted,decision=choose_recovery(region,candidates,numeric_views)
            records.append(RecoveryRecord(region.id,trigger,tuple(candidates),adopted,decision,
                                          numeric_views,perf_counter()-started))
        return apply_recoveries(regions,records)
    finally:
        page.close(); document.close()
