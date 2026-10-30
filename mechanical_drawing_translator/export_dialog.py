# mechanical_drawing_translator/export_dialog.py
"""中文标注导出范围、格式和目标路径对话框。"""

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
)


class ExportDialog(QDialog):
    """只收集导出意图，实际写盘由后台导出服务完成。"""

    # 这里只收集导出意图
    # 按钮确认不等于写盘成功
    # 单页图不显示全部页选项
    # PDF输入只提供PDF输出
    # PDF默认选择全部页面
    # 整份导出仍逐页验审核资格
    # 图片提供PNG/PDF/JPG/BMP
    # CMYK优先走PDF保留通道
    # 探测失败交给后续验证
    # 路径默认指向新副本名称
    # 不允许在这里绕过源保护
    # 格式说明紧随实际选择
    # 可访问名称供辅助技术识别
    # 具体文件能力由服务拒绝
    def __init__(self, source_path: Path, page_count: int, parent=None):
        super().__init__(parent)
        self.source_path = Path(source_path)
        self.setWindowTitle("导出中文标注图纸")
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("当前页", False)
        if page_count > 1:
            self.scope_combo.addItem("全部页面", True)
        self.format_combo = QComboBox()
        if self.source_path.suffix.lower() == ".pdf":
            self.format_combo.addItem("PDF", "PDF")
            self.scope_combo.setCurrentIndex(self.scope_combo.count() - 1)
        else:
            for value in ("PNG", "PDF", "JPG", "BMP"):
                self.format_combo.addItem(value, value)
            # A CMYK drawing has a native-channel PDF path. Select it before
            # presenting formats so the default cannot silently use RGB preview.
            from PIL import Image
            try:
                with Image.open(self.source_path) as image:
                    if image.mode == 'CMYK':
                        self.format_combo.setCurrentIndex(self.format_combo.findData('PDF'))
            except (OSError, ValueError):
                pass  # Opening/validation remains the document loader's job.
        destination_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        destination_row.addWidget(self.path_edit, 1)
        destination_row.addWidget(browse)
        form.addRow("导出范围", self.scope_combo)
        form.addRow("输出格式", self.format_combo)
        form.addRow("保存到", destination_row)
        root.addLayout(form)
        self.fidelity_notice = QLabel()
        self.fidelity_notice.setWordWrap(True)
        self.fidelity_notice.setAccessibleName('输出格式保真说明')
        root.addWidget(self.fidelity_notice)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.format_combo.currentIndexChanged.connect(self._set_default_path)
        self._set_default_path()

    # 读取itemData标准格式名
    # 显示标签不承担业务协议
    @property
    def output_format(self) -> str:
        return str(self.format_combo.currentData())

    # 范围使用布尔itemData
    # 不从中文显示文字推断
    # 页索引仍由窗口构造
    @property
    def all_pages(self) -> bool:
        return bool(self.scope_combo.currentData())

    # 先清除首尾空白
    # 最终路径安全由服务检查
    @property
    def output_path(self) -> Path:
        return Path(self.path_edit.text().strip())

    # 每次换格式重置默认路径
    # JPG使用常见jpg后缀
    # 输出名称加zh_annotated
    # PNG无损不等于所有输入无转换
    # JPEG修改后仍是有损重编码
    # BMP不能保留完整ICC/EXIF
    # PDF源保持页面对象
    # 图片PDF按每像素一点布局
    # 图片DPI不冒充PDF物理尺寸
    # 特殊PDF组合可能安全拒绝
    def _set_default_path(self) -> None:
        suffix = ".jpg" if self.output_format == "JPG" else f".{self.output_format.lower()}"
        self.path_edit.setText(str(
            self.source_path.with_name(f"{self.source_path.stem}_zh_annotated{suffix}"),
        ))
        notices = {
            'PNG':'无损编码，保留 RGB/透明度与 ICC。灰度或调色板添加红字需扩展为 RGB；CMYK 请选 PDF。',
            'JPG':'修改后 JPEG 为有损编码，可能改变未标注像素；透明度合成白底。建议选择 PNG 或 PDF；CMYK 标注请选 PDF。',
            'BMP':'BMP 兼容输出：透明度合成白底，ICC 转换至 sRGB 后不保存配置，EXIF 不保留。CMYK 请选 PDF。',
            'PDF':('保留原 PDF 页面对象并添加矢量中文；特殊旋转与原生注释组合可能拒绝。'
                   if self.source_path.suffix.lower()=='.pdf' else
                   '保留原始图像通道与 ICC，中文为矢量；常规 JPEG 保留原编码。图片 PDF 使用每像素一点的页面坐标，打印物理尺寸不沿用 DPI。'),
        }
        self.fidelity_notice.setText(notices[self.output_format])

    # 文件筛选匹配当前格式
    # 取消浏览保留原路径
    # 用户选路径不立即创建文件
    def _browse(self) -> None:
        filters = {
            "PDF": "PDF 文件 (*.pdf)",
            "PNG": "PNG 图片 (*.png)",
            "JPG": "JPEG 图片 (*.jpg *.jpeg)",
            "BMP": "BMP 图片 (*.bmp)",
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "保存中文标注副本", str(self.output_path), filters[self.output_format],
        )
        if path:
            self.path_edit.setText(path)

    # 空路径保持对话框打开
    # 非空仅返回用户意图
    # 审核和覆盖风险随后核验
    def accept(self) -> None:
        if self.path_edit.text().strip():
            super().accept()
