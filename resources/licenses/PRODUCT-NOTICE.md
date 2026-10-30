# MechanicalDrawingTranslator offline distribution notices

This Onedir distribution contains dynamically linked third-party components.
Keep the complete `_internal` directory with the executable. The editable
application Python source and build specification are supplied separately.

Preserve all distribution metadata and their `licenses` directories, the font
OFL notice, OCR model notices, and the notices in this directory. Python is
distributed under its PSF terms (included in `python/LICENSE.txt`).

Components: PySide6/Qt/Shiboken6 (wheel metadata declares LGPL/GPL alternatives),
RapidOCR and the bundled PaddleOCR weights (Apache-2.0, separate attributions in
the wheel's MODEL_LICENSES.md), ONNX Runtime, NumPy, OpenCV, Pillow, pdfminer.six,
pdfplumber, pypdfium2/PDFium, pypdf, CTranslate2, SentencePiece, OmegaConf and
their transitive dependencies. Installed versions and license metadata are
included in the adjacent `*.dist-info` directories.

Argos en-zh 1.9 weights: CC-BY-4.0 as declared by the original package; attribution,
original artifact identity and source are in translation-runtime-SOURCES.md.
Noto Sans SC font: SIL OFL, with font source notice beside the OTF.

QuickMT and Semantic Edit experimental model assets are not included.
Runtime inference resolves local resource paths and does not need model downloads.

Qt license text sources (obtained at package preparation time):
https://github.com/qt/qtbase/blob/dev/LICENSES/LGPL-3.0-only.txt
https://github.com/qt/qtbase/blob/dev/LICENSES/GPL-3.0-only.txt

This notice inventories included evidence; it is not a completed legal or
redistribution compliance certification. Review the applicable terms and
corresponding source/relinking obligations before commercial redistribution.
