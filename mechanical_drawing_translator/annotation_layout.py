# mechanical_drawing_translator/annotation_layout.py
"""在原图像素或 PDF 显示点坐标中计算共享标注布局。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import cos, isclose, radians, sin
import re

from PySide6.QtGui import QFontMetricsF

from .annotation_models import (
    AnnotationItem, AnnotationLayout, MaskDecision, PageGeometry, Polygon,
)
from .font_resources import annotation_font, ensure_font_supports, load_annotation_font
from .models import TextBounds
from .translation_results import STAGE1_PROVENANCE, TranslationResult
from .translation_targets import TargetEntry


# 布局引擎只计算源坐标中的几何，不把当前窗口缩放写进数据。
# PDF 页使用显示方向的 point，图片使用完成 EXIF 定向后的像素。
# 预览和正式布局共享字体、换行与旋转规则，但入口资格不同。
# layout_preview 接受带来源的机器预览；layout 保留正式审核门禁。
# 此处的文本框不是原始 OCR 框的替代品，原框保存在 source_polygons。
# 候选失败以无绘制行和待布局状态表达，不能截断中文伪造成功。
# 遮挡申请与文字放置分开记录，申请本身不代表背景安全。
# 保存对象后仍须在绘制、恢复和导出边界复核几何与资格。
class AnnotationLayoutEngine:
    def layout_preview(self, page, entries, results, *, source_image,
                       mask_requested=False, obstacles=(),mask_background=None,
                       verified_mask_groups=frozenset(),display_mode=None,
                       supplemental_masks=None,supplemental_reasons=None):
        """Explicit machine-capable preview; export layout() keeps its gate."""
        # 延迟导入避免预览模块与本引擎之间的顶层循环依赖。
        # 所有预览分支都经过同一分派器，不能直接调用正式 layout 冒充预览。
        # 背景和 verified 分组是证据输入，最终仍由预览模块逐项重新核验。
        # 补充栅格参数保留兼容接口，不能据此赋予 OCR 自动擦除权限。
        from .annotation_preview import create_preview_layout
        return create_preview_layout(self, page, entries, results, source_image,
                                     mask_requested, obstacles,mask_background,verified_mask_groups,display_mode,
                                     supplemental_masks,supplemental_reasons)

    def __init__(self, *, min_font_size: float = 6.0, max_font_size: float = 18.0):
        # 构造时拒绝倒置字号区间，避免后续缩字号循环永远找不到终点。
        # 默认值是排版约定，不表示适用于每种 PDF/图片的专业打印字号。
        # 不同源单位下同一数值的物理大小不同，调用方应使用正确 PageGeometry。
        # 更高层的六单位可读门禁仍会在紧字形和正式导出时复核。
        if min_font_size <= 0 or max_font_size < min_font_size:
            raise ValueError("annotation font-size range is invalid")
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size

    def layout(
        self,
        page: PageGeometry,
        entries: Sequence[TargetEntry],
        results: Mapping[str, TranslationResult],
        *,
        mask_requested: bool,
    ) -> AnnotationLayout:
        # 缺少翻译结果的目标不产生正式标注，原图依旧保留。
        # 逐项同时绑定目标、结果和页面身份，防止切页缓存错贴译文。
        # 审核来源在计算坐标前检查，避免生成可误用为正式输出的机器布局。
        # 这里不提升 provenance，人工审核记录必须由上游真实提供。
        page_entries = [entry for entry in entries if entry.group_id in results]
        for entry in page_entries:
            result = results[entry.group_id]
            if result.provenance not in STAGE1_PROVENANCE:
                raise ValueError("机器译文尚未完成人工审核，不能生成标注布局")
            if entry.document_id != page.document_id or entry.page_index != page.page_index:
                raise ValueError("translation target does not belong to this page")
            if result.document_id != page.document_id or result.page_index != page.page_index:
                raise ValueError("translation result does not belong to this page")
        items: list[AnnotationItem] = []
        # 原文框一开始就是障碍，包括没有可用译文的目标。
        # 放置成功的中文随后加入 occupied，使后续项不覆盖先前标注。
        # 顺序贪心是此基础算法的特征，双语联合求解在另一个模块补充。
        # 不要把遮挡后看似空白的英文位置当成双语排版的自由空间。
        occupied = [entry.bounds for entry in entries]
        for entry in page_entries:
            result = results[entry.group_id]
            ensure_font_supports(result.translated_text)
            source_bounds = entry.bounds
            rotation = self._dominant_rotation(entry)
            # 初始字号按原框高度估计，并限制在引擎的可读区间。
            # 这是源坐标字号，不随屏幕 DPI 或窗口缩放变化。
            # 宽度上限兼顾短字段和较长说明，避免一行横跨大半张图。
            # 适配只改变完整译文的换行和字号，不改变译文字面。
            # 若最小字号仍不满足最大宽高，_fit_text 明确抛出异常。
            initial_size = min(self.max_font_size, max(
                self.min_font_size,
                (source_bounds.max_y - source_bounds.min_y) * 0.86,
            ))
            max_width = min(
                page.width * 0.42,
                max(source_bounds.max_x - source_bounds.min_x, page.width * 0.18),
            )
            lines, font_size, text_width, text_height = self._fit_text(
                result.translated_text, max_width, page.height * 0.42, initial_size,
            )
            # 复制翻译警告而不修改 TranslationResult，排版诊断与翻译来源各自保留。
            # 字体可排不代表工程语义正确，因此不能清掉原翻译风险提示。
            # fallback 单独记录遮挡拒绝后的文字位置，不能覆盖原始 source_bounds。
            # placement 初始化为 inline，失败分支必须显式改为待布局并清行。
            warnings = list(result.warnings)
            fallback_text_bounds = None
            placement = 'inline'
            # 正式基础布局先记录待分析遮挡；背景安全由后续分析判定。
            # 原位文字同时保留邻近备用框，以便拒绝遮挡时仍可展示中文。
            # 未申请遮挡则只搜索邻近位置，原英文不需要被移除。
            # 当前产品的双语可选遮挡另走 apply_bilingual_masks，位置保持独立。
            # 不要把这个历史基础分支视作当前第三种产品模式入口。
            if mask_requested:
                text_bounds = self._clamped_bounds(
                    source_bounds.min_x, source_bounds.min_y,
                    text_width, text_height, page,
                )
                fallback_text_bounds, _ = self._place_adjacent(
                    source_bounds, text_width, text_height, page, occupied,
                )
                decision = MaskDecision.PENDING_ANALYSIS
                masks = self._source_polygons(entry)
            else:
                placed = self._fit_local(result.translated_text, source_bounds,
                                         rotation, page, occupied, initial_size)
                if placed is None:
                    text_bounds = source_bounds
                    lines = ()
                    placement = 'needs_layout_review'
                    warnings.append('needs_layout_review：邻近区域无法安全放置，请人工调整')
                else:
                    text_bounds, lines, font_size = placed
                decision = MaskDecision.NOT_REQUESTED
                masks = ()
            # 平移未旋转的框，使最终旋转包围框不会越过纸张。
            # 备用位置必须使用相同角度校正，否则遮挡失败会产生越界跳转。
            # 记录占用使用实际可用的备用框或当前框，保持后续布局一致。
            # 这里的页面钳制不是背景安全证明，仍有独立碰撞检查。
            # AnnotationItem 同时保留原文、原区域与翻译来源，便于追溯。
            text_bounds = self._clamp_rotated_footprint(text_bounds, rotation, page)
            if fallback_text_bounds is not None:
                fallback_text_bounds = self._clamp_rotated_footprint(
                    fallback_text_bounds, rotation, page,
                )
            occupied.append(fallback_text_bounds or text_bounds)
            items.append(AnnotationItem(
                group_id=entry.group_id,
                region_ids=entry.region_ids,
                source_text=entry.source_text,
                translated_text=result.translated_text,
                provenance=result.provenance,
                source_polygons=self._source_polygons(entry),
                text_bounds=text_bounds,
                font_size=font_size,
                lines=lines,
                rotation_degrees=rotation,
                mask_polygons=masks,
                mask_decision=decision,
                warnings=tuple(warnings),
                fallback_text_bounds=fallback_text_bounds,
                placement=placement,
            ))
        return AnnotationLayout(
            page.document_id, page.page_index, page.width, page.height,
            page.coordinate_unit, load_annotation_font(), mask_requested, tuple(items),
        )

    def _fit_local(self, text, source, rotation, page, occupied, initial_size, blank=None):
        """Search only an anchor's neighborhood; never create a remote overflow area.

        Candidate rectangles represent the rotated footprint. This avoids moving
        a rotated label after collision testing, which invalidates that test.
        """
        from .annotation_preview import footprint, _pad
        # 局部搜索只有有限字号、宽度、方向和距离，成本不会随像素面积爆炸。
        # 旋转先作用于候选 footprint，再拿它与障碍比较。
        # 如果先用水平框通过碰撞后再旋转，角落可能碰到尺寸线。
        # dict.fromkeys 保留尝试顺序同时去掉重复字号。
        # blank 为空时仅有框级几何判断，预览通常提供真实像素空白检查。
        sizes = list(dict.fromkeys((initial_size, max(self.min_font_size, initial_size - 2),
                                   self.min_font_size)))
        source_width = source.max_x - source.min_x
        max_width = min(page.width * .30, max(source_width, page.width * .12))
        best = None
        for size in sizes:
            for width in (max_width, min(page.width * .38, max_width * 1.5), max_width * .70):
                try:
                    lines, fitted_size, w, h = self._fit_text(text, width, page.height * .25, size)
                except ValueError:
                    continue
                # 原点框只用于计算旋转后的宽高，不改变源锚点。
                # rw/rh 是轴对齐包围框尺寸，w/h 是文字自身未旋转尺寸。
                # 候选坐标先描述 footprint 左上角，再反推出文字框中心。
                # 这两个框不能混用，否则 90 度文字会沿错误方向避让。
                origin = TextBounds(0, 0, w, h)
                rotated = footprint(origin, rotation)
                rw = rotated.max_x - rotated.min_x
                rh = rotated.max_y - rotated.min_y
                candidates = []
                gap = 3.0
                # 有限距离环在原文四侧逐步尝试，先近后远只影响候选集合。
                # 最后仍跨字号统一排序，不保证先发现的候选就是最终结果。
                # 空间距离按框边衡量，允许长原框某端的真正邻近空隙。
                # 四十单位上限之外不创建远端溢出标签，以免中文失去源关联。
                for distance in (gap, gap + 8, gap + 20, gap + 36):
                    for x, y in ((source.min_x, source.max_y + distance),
                                 (source.min_x, source.min_y - rh - distance),
                                 (source.max_x + distance, source.min_y),
                                 (source.min_x - rw - distance, source.min_y)):
                        # Clamp within the actual page, then recheck collisions.
                        x = max(2.0, min(x, page.width - rw - 2))
                        y = max(2.0, min(y, page.height - rh - 2))
                        b = TextBounds(x + rw/2 - w/2, y + rh/2 - h/2,
                                       x + rw/2 + w/2, y + rh/2 + h/2)
                        # 一单位保护边留给抗锯齿和视觉间距，不只检查黑色字心。
                        # 先钳制再重新算 footprint，避免钳制导致候选贴到另一项上。
                        # 页面内部仍留边距；存在图框时 blank.clear 还会收紧边界。
                        # 框级碰撞优先拒绝，像素检查再拒绝未登记的线稿或淡色墨迹。
                        # 距离按两个矩形间的空隙计算，而不是中心点距离。
                        f = _pad(footprint(b, rotation), 1)
                        if not (f.min_x >= 1 and f.min_y >= 1 and
                                f.max_x <= page.width-1 and f.max_y <= page.height-1):
                            continue
                        if any(self._overlap_area(f, other) > 0 for other in occupied):
                            continue
                        if blank is not None and not blank.clear(f):
                            continue
                        dx=max(source.min_x-f.max_x, f.min_x-source.max_x, 0)
                        dy=max(source.min_y-f.max_y, f.min_y-source.max_y, 0)
                        if (dx*dx+dy*dy)**.5 > 40:
                            continue
                        candidates.append((dx*dx+dy*dy, b))
                if candidates:
                    distance, bounds = min(candidates, key=lambda value:value[0])
                    # Proximity is the user's primary association rule. Do not
                    # stop at a distant large-font rectangle if a still-readable
                    # smaller/font-wrapped candidate fits beside the English.
                    # 排序首先保持中文与对应英文邻近，减少读者误配。
                    # 同距离才优先大字号和较少换行，不能为大字把译文移到远处。
                    # 跨字号比较保留全局最好候选，不在第一个可用大字号处提前停止。
                    # 没有合法候选时返回 None，由调用者保留原图并提示待布局。
                    rank = (distance, -fitted_size, len(lines))
                    if best is None or rank < best[0]:
                        best = (rank, bounds, lines, fitted_size)
        return (best[1],best[2],best[3]) if best is not None else None

    def validate_placement(self, page, item, *, source_image, obstacles=(), occupied=()):
        """Return safety failures for manual labels using final source coordinates."""
        from .annotation_preview import footprint, BlankSpace, _pad
        # 人工移动也使用最终源坐标复验，拖动操作不豁免安全规则。
        # 页面、背景和其它标注分别给出原因，便于界面解释拒绝。
        # 同 group 的旧框不作为自己的障碍，其它 inline 标注仍参与。
        # 此方法返回诊断而不写回 item，调用者决定是否接受人工事务。
        warnings=[]
        f=_pad(footprint(item.text_bounds,item.rotation_degrees),1)
        if (f.min_x < 1 or f.min_y < 1 or f.max_x > page.width-1 or f.max_y > page.height-1):
            warnings.append('outside_effective_page')
        if not BlankSpace(source_image,page).clear(f):
            warnings.append('background_ink_collision')
        bounds=[getattr(o,'bounds',o) for o in obstacles]
        bounds.extend(footprint(o.text_bounds,o.rotation_degrees) for o in occupied
                      if o.group_id != item.group_id and o.placement == 'inline')
        if any(self._overlap_area(f,b)>0 for b in bounds):
            warnings.append('annotation_or_source_collision')
        return tuple(warnings)

    def _fit_text(
        self, text: str, max_width: float, max_height: float, initial_size: float,
    ) -> tuple[tuple[str, ...], float, float, float]:
        # 字体度量来自最终绘制使用的同一应用字体。
        # lineSpacing 覆盖基线间距，horizontalAdvance 覆盖每行占用宽度。
        # 宽高必须同时成立，只有高度满足并不代表不会跨过工程线。
        # 逐次减一在最小字号处终止，禁止静默压成不可读的小字。
        # 返回 lines 是完整结果，绘制器不应再次进行另一套换行。
        size = initial_size
        while True:
            # 换行宽度与字体实际 horizontalAdvance 一致，不能用字符数近似。
            # 中文、拉丁标识和工程符号字宽不同，等字符计数会造成实际溢出。
            # 行高乘行数包含显式空行，不用文字墨迹高度替代旧版基线合同。
            # 紧字形路径另由 tight_text_fit 使用实际 ink，并保存其专属 origin。
            font = annotation_font(size)
            metrics = QFontMetricsF(font)
            lines = self._wrap(text, metrics, max_width)
            line_height = metrics.lineSpacing()
            width = max(metrics.horizontalAdvance(line) for line in lines)
            height = line_height * len(lines)
            if height <= max_height and width <= max_width:
                return lines, max(size, self.min_font_size), width, height
            if size <= self.min_font_size or isclose(
                size, self.min_font_size, abs_tol=0.01,
            ):
                raise ValueError("中文译文过长，最小字号下仍无法放入页面；请人工缩短或拆分")
            size = max(self.min_font_size, size - 1)

    @staticmethod
    def _wrap(text: str, metrics: QFontMetricsF, max_width: float) -> tuple[str, ...]:
        lines: list[str] = []
        current = ""
        # 显式段落换行即使前行尚未满也保留，避免把条件与说明合为一行。
        # 行首/行尾空白在自然折行处清理，不改变非空白工程字面。
        # 空文本仍返回一行的历史合同，由上层可用性检查决定是否绘制。
        # 宽度不足容纳单个工程 token 时，不拆数字或型号，交外层缩号或拒绝。
        for paragraph_index, paragraph in enumerate(text.splitlines() or [""]):
            # Keep engineering identifiers, decimals and their units intact.
            # Closing punctuation stays with the preceding line; if this makes
            # a line too wide, _fit_text retries at a readable smaller size.
            # 识别串、小数和常见单位先组成 token，防止拆成两行歧义。
            # 例如尺寸值及 mm 应作为一个排版单元保留。
            # 中文与普通标点仍可逐字符折行，显式换行保留段落边界。
            # 右侧闭合标点跟随前文，超宽交给外层缩字号处理。
            # 正则仅服务排版，不用它判断工程内容是否正确或允许翻译。
            # 不能用删空格、删单位或截去尾句解决宽度不足。
            tokens = re.findall(r'[±ΦØ⌀]?[A-Za-z0-9][A-Za-z0-9_./°±×+\-]*(?:[ \t]+(?:mm|cm|kg|MPa|N|m|in)\b)?|.', paragraph)
            for character in tokens:
                proposed = current + character
                if (current and metrics.horizontalAdvance(proposed) > max_width
                        and character not in '，。；：！？、）》】%'):
                    lines.append(current.rstrip())
                    current = character.lstrip()
                else:
                    current = proposed
            if paragraph_index < len(text.splitlines()) - 1:
                lines.append(current)
                current = ""
        if current or not lines:
            lines.append(current)
        return tuple(lines)

    @staticmethod
    def _source_polygons(entry: TargetEntry) -> tuple[Polygon, ...]:
        # 原始 polygon 优先，只有缺少多边形时才用原 bounds 构造矩形。
        # 一个翻译组可能关联多个原始区域，不能只保留组的并集框。
        # 这些多边形用来定位和候选遮挡，不等于已验证的擦除范围。
        # 返回 tuple 让布局可以安全跨预览、保存与导出共享。
        polygons = []
        for region in entry.regions:
            polygons.append(region.polygon or (
                (region.bounds.min_x, region.bounds.min_y),
                (region.bounds.max_x, region.bounds.min_y),
                (region.bounds.max_x, region.bounds.max_y),
                (region.bounds.min_x, region.bounds.max_y),
            ))
        return tuple(polygons)

    @staticmethod
    def _dominant_rotation(entry: TargetEntry) -> float:
        # 多区域组以面积最大的原区域决定排版主方向。
        # 此规则不平均角度，避免 0 度与 180 度平均成不相关的 90 度。
        # 它只是展示方向启发式，不改任何原区域旋转或 OCR 坐标。
        # 调用者应保证目标包含区域，否则这里没有可靠方向来源。
        weighted = sorted(
            entry.regions,
            key=lambda region: (region.bounds.max_x - region.bounds.min_x)
            * (region.bounds.max_y - region.bounds.min_y),
            reverse=True,
        )
        return weighted[0].rotation_degrees

    @staticmethod
    def _clamped_bounds(
        x: float, y: float, width: float, height: float, page: PageGeometry,
    ) -> TextBounds:
        # 框尺寸先收敛到纸张，再平移左上角，避免负的剩余空间。
        # 钳制只保证轴对齐框在页内，旋转框仍需下一层检查。
        # 不要把显示视口的宽高传入 page，否则持久化位置会随窗口变化。
        # 此辅助函数不会验证框内是否有原图墨迹。
        width = min(width, page.width)
        height = min(height, page.height)
        x = min(max(0.0, x), page.width - width)
        y = min(max(0.0, y), page.height - height)
        return TextBounds(x, y, x + width, y + height)

    def _place_adjacent(
        self, source: TextBounds, width: float, height: float,
        page: PageGeometry, occupied: Sequence[TextBounds],
    ) -> tuple[TextBounds, bool]:
        # 邻近备用位置只尝试下、上、右、左四个方向。
        # 先接受完全不相交位置；全冲突时返回交叠最小者并附带失败标志。
        # 该历史备用算法不提供像素空白保证，调用方不能忽略安全分析。
        # 当前双语联合布局拥有更完整候选和事务求解，不在这里追加远端区。
        gap = 2.0
        candidates = (
            (source.min_x, source.max_y + gap),
            (source.min_x, source.min_y - height - gap),
            (source.max_x + gap, source.min_y),
            (source.min_x - width - gap, source.min_y),
        )
        clamped = [self._clamped_bounds(x, y, width, height, page) for x, y in candidates]
        for candidate in clamped:
            if not any(self._overlap_area(candidate, other) > 0 for other in occupied):
                return candidate, False
        best = min(clamped, key=lambda candidate: sum(
            self._overlap_area(candidate, other) for other in occupied
        ))
        return best, True

    @staticmethod
    def _overlap_area(left: TextBounds, right: TextBounds) -> float:
        # 负交集截为零，边缘接触不产生正面积。
        # 此处使用连续源坐标而不是整数像素，避免缩放导致判定抖动。
        # 旋转项必须先转为 footprint 再使用这个轴对齐交叠函数。
        return max(0.0, min(left.max_x, right.max_x) - max(left.min_x, right.min_x)) * max(
            0.0, min(left.max_y, right.max_y) - max(left.min_y, right.min_y)
        )

    @staticmethod
    def _clamp_rotated_footprint(
        bounds: TextBounds, angle_degrees: float, page: PageGeometry,
    ) -> TextBounds:
        """平移文字框，使绕中心旋转后的四角仍位于页面内。"""
        if isclose(angle_degrees % 360, 0.0, abs_tol=0.01):
            return bounds
        # 旋转围绕文字框中心，与共享 painter 的变换顺序一致。
        # 枚举四角后求极值，适用于非直角的包围框检查。
        # 只平移文字框而不改变其宽高、旋转和字号。
        # 页面小于旋转 footprint 时此辅助钳制不能创造可行空间。
        # 最终绘制和导出仍会拒绝越界，不能把平移等同于完整验证。
        # 返回框仍是未旋转框，后续绘制器会按原角度旋转它。
        center_x = (bounds.min_x + bounds.max_x) / 2
        center_y = (bounds.min_y + bounds.max_y) / 2
        angle = radians(angle_degrees)
        rotated = []
        for x in (bounds.min_x, bounds.max_x):
            for y in (bounds.min_y, bounds.max_y):
                dx, dy = x - center_x, y - center_y
                rotated.append((center_x + dx * cos(angle) - dy * sin(angle),
                                center_y + dx * sin(angle) + dy * cos(angle)))
        # 四角极值描述旋转后轴对齐包络，不是更精细的字形碰撞多边形。
        # 保守包络可能拒绝一些其实没有笔画的角落空隙，优先保护线稿。
        # 只作 X/Y 平移不会改变该包络大小；超大旋转框仍由后续边界拒绝。
        # 原文字多边形不参与这里的移动，源关联继续停留在原锚点。
        min_x = min(x for x, _ in rotated)
        max_x = max(x for x, _ in rotated)
        min_y = min(y for _, y in rotated)
        max_y = max(y for _, y in rotated)
        shift_x = -min_x if min_x < 0 else page.width - max_x if max_x > page.width else 0
        shift_y = -min_y if min_y < 0 else page.height - max_y if max_y > page.height else 0
        return TextBounds(
            bounds.min_x + shift_x, bounds.min_y + shift_y,
            bounds.max_x + shift_x, bounds.max_y + shift_y,
        )
