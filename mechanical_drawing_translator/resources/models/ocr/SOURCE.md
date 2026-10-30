# OCR资源来源与迁移

2026-10-30从本机已安装的RapidOCR 3.10.0 wheel复制三个既有ONNX，未下载、重新训练、转换或修改权重。文件身份见主项目resources/RESOURCE_MANIFEST.json与工作区迁移记录。

模型是Baidu/PaddleOCR权利方的权重转换产物；本地MODEL_LICENSES.md逐件列出上游、来源URL、SHA与Apache-2.0声明，LICENSE保留RapidOCR版权及Apache正文。只适用于清单中的三个模型。RapidOCR的config.yaml/default_models.yaml及Python推理实现仍由固定版本依赖提供，未来打包须collect_all(rapidocr)，不可只带ONNX。
