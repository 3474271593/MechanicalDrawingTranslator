# mechanical_drawing_translator/layout_identity.py
"""Generated geometry identity, separate from model/translation provenance."""
# 几何身份独立于模型、知识资源和翻译来源版本。
# 自动布局缓存只有与当前身份一致才可直接复用。
# 安全规则变化需使旧自动几何失效，尤其不能继承旧 OCR 擦除资格。
# 人工位置和原 OCR/SourceSpan 应保留追溯，再按当前规则复验。
# 此常量不表示旧 chinese 数据被删除；恢复流程负责迁移到双语。
# 仅补注释不更改本值，避免无业务变化却触发全量重新布局。
# 版本字符串不是签名或真实证据，不能替代源 SHA 与独立背景核验。
LAYOUT_ALGORITHM_VERSION='bilingual-optional-mask-2026-10-29-v5'
