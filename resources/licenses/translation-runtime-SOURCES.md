# Offline translation runtime sources

The D6 offline translation service redistributes the following pinned components:

| Component | Version/artifact | License | Source |
| --- | --- | --- | --- |
| Argos English-Chinese model | `translate-en_zh-1_9.argosmodel`, package 1.9 | Model package declares CC-BY-4.0 | <https://github.com/argosopentech/argospm-index> |
| CTranslate2 | 4.8.3 | MIT | <https://github.com/OpenNMT/CTranslate2> |
| SentencePiece | 0.2.1 | Apache-2.0 | <https://github.com/google/sentencepiece> |

The original Argos package SHA-256 is `433e7c4f034d87fbe2353161e05f18646d7999452f801a4e1f0378522b9850ab`. File-level identities for the redistributed model are recorded in the packaged `model-manifest.json`.

The application performs no runtime model download. The model, tokenizer, Python wheels/DLLs, notices, and license files must be included in the offline Windows Onedir distribution.
