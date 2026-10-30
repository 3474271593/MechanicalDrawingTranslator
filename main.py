# main.py
"""启动本地 PySide6 桌面应用。"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path
from mechanical_drawing_translator.resource_paths import project_resource

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
from PySide6.QtWidgets import QApplication

from mechanical_drawing_translator.main_window import MainWindow
from mechanical_drawing_translator.translation_worker import make_translation_service
from mechanical_drawing_translator.ocr_worker import make_ocr_service


# 参数仅声明可选能力
# 实际生效看runtime_options
# stable是main直启默认
# 正式入口由launch选择product
# 候选能力都有显式参数
# 自定义知识包还需可信SHA
# 审核资源不等于导出批准
# OCR试验与翻译试验分开
# stage3包含CPU优化配置
# ORT缓存目录单独指定
# 该目录不是模型目录
# 不在解析阶段下载资源
# Qt参数留给QApplication
def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MechanicalDrawingTranslator")
    parser.add_argument('--runtime-profile', choices=('stable', 'd7-candidate', 'd7d8-candidate',
                                                     'd7d8-reviewed-candidate'),
                        default='stable', help='隔离的整合候选配置；默认仍为稳定版本')
    parser.add_argument('--knowledge-mode', choices=('off','shadow','candidate'), default='off',
                        help='V2知识运行方式；off保留原日常数据调用')
    parser.add_argument('--knowledge-root', type=Path, help='自定义V2运行包目录，须同时指定可信哈希')
    parser.add_argument('--knowledge-manifest-sha256', help='自定义V2运行包manifest可信SHA256')
    parser.add_argument('--enable-v2-quality-optimization', action='store_true',
                        help='显式开启V2质量收尾试验；须candidate模式')
    parser.add_argument('--enable-reviewed94',action='store_true',
                        help='启用用户94条审核增量候选；须V2 candidate，不改变PDF导出资格')
    parser.add_argument('--enable-integrated-quality',action='store_true',
                        help='启用综合质量候选的混框保留、分句身份和受控工序组合；须V2 candidate')
    parser.add_argument('--enable-professional-quality',action='store_true',
                        help='启用有限机械语义生成与关系校验候选；须V2 candidate，机器预览不改变导出资格')
    parser.add_argument('--enable-convergence-quality',action='store_true',help='显式启用收敛候选')
    parser.add_argument('--enable-argos-core',action='store_true',help='显式启用Argos专业约束及同请求并行加载候选')
    parser.add_argument('--enable-final-optimization',action='store_true',help='显式启用专业候选/关系核验与OCR优化图缓存；须ArgosCore')
    parser.add_argument('--product-core', action='store_true', help='正式产品的可信知识组合及安全校验；排除实验推理和ORT缓存')
    parser.add_argument('--enable-semantic-edit',action='store_true',help='显式启用同次模型源对齐术语修订；须Final Optimization')
    parser.add_argument('--ocr-optimized-cache-dir',type=Path,help='独立候选OCR优化图缓存；不写源模型/原图')
    parser.add_argument('--enable-reference-quality',action='store_true',
                        help='启用已有知识结构复用与附加关系校验；须professional candidate，未宣称95%验收')
    parser.add_argument(
        "--enable-d7e-trial", action="store_true",
        help="显式启用D7-B至D7-E实验增强；机器译文仍需人工审核",
    )
    parser.add_argument(
        "--enable-ocr-3b-trial", action="store_true",
        help="显式启用OCR-3B语义句界试运行；保留原始OCR与共享区域定位",
    )
    parser.add_argument('--enable-d7e-marked-rules-trial', action='store_true',
                        help='显式启用前导图纸星号保护与审核正文复用')
    parser.add_argument('--enable-d7f-trial', action='store_true',
                        help='启用图签字段和工程量术语试运行，仍需人工审核')
    parser.add_argument('--enable-d7g-trial',action='store_true',help='启用审核子句工序关系组合试运行')
    parser.add_argument('--enable-d8a-trial', action='store_true',
                        help='启用D8-A图签语境与编号安全校验试运行，机器译文仍需人工审核')
    parser.add_argument('--enable-d8b-trial', action='store_true',
                        help='启用D8-B专业表达知识治理试运行，机器译文仍需人工审核')
    parser.add_argument('--enable-engineering-symbol-risk-trial', action='store_true',
                        help='启用工程符号OCR高风险提示与安全回退试运行')
    parser.add_argument('--enable-d8c-semantic-trial', action='store_true',
                        help='启用D8-C审核短语句界恢复试运行，机器译文仍需人工审核')
    parser.add_argument('--enable-d8c-ocr-verification-trial', action='store_true',
                        help='启用D8-C局部图像工程字符核验试运行，保留原始OCR')
    parser.add_argument("--enable-ocr-3c-trial",action="store_true",
                        help="显式启用OCR-3C局部图像复查，保留原始OCR与候选证据")
    parser.add_argument('--enable-ocr-stage2-trial',action='store_true',
                        help='试运行8线程/CPU arena及并行会话初始化；内存占用更高')
    parser.add_argument('--enable-ocr-stage3-trial',action='store_true',
                        help='统一OCR候选：包含第二轮加速并减少动态形状内存驻留')
    return parser


# 纯参数解析供集成复用
# 测试可传argv隔离进程
# 返回值尚未创建窗口
def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    return _argument_parser().parse_args(argv)


# 先验证候选的父配置
# 缺少父能力直接拒绝
# 不能让孤立开关悄悄降级
# 窗口创建后配置固定
# 翻译和OCR分别注入工厂
# 主线程不持有推理模型
# d7/d8按profile整体展开
# 显式标志还能单独启用
# stage3同时设置两项OCR选项
# compact控制动态形状内存
# 审核资源位于项目data目录
# 路径从当前源码位置推导
# product只加产品核心标志
# argos-core触发加载预备
# final才设置ORT缓存目录
# 缓存优先显式参数
# 其次读取环境变量
# 最后写用户数据目录
# 资源目录不接收运行缓存
# knowledge off不传V2配置
# 自定义manifest保持显式
# 这些开关不改变机器来源
# 质量验收不由开关名证明
def runtime_options(args: argparse.Namespace) -> dict:
    """Resolve one immutable-at-window-creation profile, preserving legacy flags."""
    if getattr(args,'enable_semantic_edit',False) and not getattr(args,'enable_final_optimization',False):
        raise ValueError('semantic edit requires final optimization')
    if getattr(args,'enable_final_optimization',False) and not args.enable_argos_core:
        raise ValueError('final optimization requires Argos core candidate parent')
    if args.enable_v2_quality_optimization and args.knowledge_mode != 'candidate':
        raise ValueError('V2 quality optimization requires --knowledge-mode candidate')
    if args.enable_reviewed94 and args.knowledge_mode != 'candidate':
        raise ValueError('reviewed94 requires --knowledge-mode candidate')
    if args.enable_integrated_quality and args.knowledge_mode != 'candidate':
        raise ValueError('integrated quality requires --knowledge-mode candidate')
    if args.enable_professional_quality and args.knowledge_mode != 'candidate':
        raise ValueError('professional quality requires --knowledge-mode candidate')
    if args.enable_reference_quality and (args.knowledge_mode!='candidate' or not args.enable_professional_quality):
        raise ValueError('reference quality requires professional candidate mode')
    if args.enable_convergence_quality and not args.enable_reference_quality:
        raise ValueError('convergence quality requires reference candidate parent')
    # profile决定旧阶段的组合资格。
    # reviewed候选也包含D7与D8。
    # 显式单项标志仍可与profile并用。
    # 解析结果保持普通数据便于核验。
    d7 = args.runtime_profile in ('d7-candidate', 'd7d8-candidate', 'd7d8-reviewed-candidate')
    d8 = args.runtime_profile in ('d7d8-candidate', 'd7d8-reviewed-candidate')
    options = {
        'translation': {
            'enable_d7e': d7 or args.enable_d7e_trial,
            'enable_marked_rules': d7 or args.enable_d7e_marked_rules_trial,
            'enable_d7f': d7 or args.enable_d7f_trial,
            'enable_d7g': d7 or args.enable_d7g_trial,
            'enable_d8a': d8 or args.enable_d8a_trial,
            'enable_d8b': d8 or args.enable_d8b_trial,
            'enable_engineering_symbol_risk': d8 or args.enable_engineering_symbol_risk_trial,
            'enable_d8c': d8 or args.enable_d8c_semantic_trial,
            'enable_d8c_ocr_verification': d8 or args.enable_d8c_ocr_verification_trial,
        },
        'ocr': {
            'enable_refinement': d8 or args.enable_ocr_3c_trial,
            'enable_engineering_verification': d8 or args.enable_d8c_ocr_verification_trial,
        },
        'enable_semantic_grouping': d8 or args.enable_ocr_3b_trial,
        'enable_d8c_semantic': d8 or args.enable_d8c_semantic_trial,
    }
    if getattr(args, 'enable_ocr_stage2_trial', False):
        options['ocr']['enable_cpu_acceleration'] = True
    if getattr(args, 'enable_ocr_stage3_trial', False):
        options['ocr']['enable_cpu_acceleration'] = True
        options['ocr']['enable_compact_memory'] = True
    if args.runtime_profile == 'd7d8-reviewed-candidate':
        options['translation']['review_overlay_path'] = (
            project_resource('data', 'user_review_overlays', 'd7d8_approved_review_v2.json')
        )
    # 综合候选同时影响OCR保留与翻译。
    # 只打开翻译端会丢失混框上下文。
    # 窗口端还必须启用语义分组。
    if args.enable_integrated_quality:
        options['translation']['enable_integrated_quality']=True
        options['ocr']['enable_integrated_quality']=True
        options['enable_integrated_quality']=True
        options['enable_semantic_grouping']=True
    if args.enable_professional_quality:
        options['translation']['enable_professional_quality']=True
        options['enable_semantic_grouping']=True
    if args.enable_reference_quality:
        options['translation']['enable_reference_quality']=True
    if args.enable_convergence_quality:
        options['translation']['enable_convergence_quality']=True
    # 产品核心依赖ArgosCore父能力。
    # 它没有顺带选择Final候选。
    # 因而不隐式开启ORT图序列化。
    # 也不会启用术语源对齐编辑实验。
    if getattr(args, 'product_core', False):
        if not args.enable_argos_core:
            raise ValueError('product core requires Argos core')
        options['translation']['enable_product_core'] = True
    if args.enable_argos_core:
        options['translation']['enable_argos_core']=True
        options['enable_runtime_preload']=True
    if getattr(args,'enable_final_optimization',False):
        options['translation']['enable_final_optimization']=True
        base=Path(os.environ.get('LOCALAPPDATA') or Path.home()/'AppData'/'Local')
        options['ocr']['optimized_graph_cache_dir']=(getattr(args,'ocr_optimized_cache_dir',None)
            or Path(os.environ.get('MDT_OCR_GRAPH_CACHE_DIR') or base/'MechanicalDrawingTranslator'/'cache'/'ort'))
    if getattr(args,'enable_semantic_edit',False):
        options['translation']['enable_semantic_edit']=True
    # V2的off/shadow/candidate分开处理。
    # shadow可保留对照而不替换结果。
    # candidate资格由配置依赖门禁保证。
    # 可信manifest不是可省略的身份信息。
    if args.knowledge_mode != 'off':
        options['translation']['knowledge_mode'] = args.knowledge_mode
        if args.enable_v2_quality_optimization:
            options['translation']['enable_v2_quality_optimization'] = True
        if args.enable_reviewed94:
            options['translation']['enable_reviewed94'] = True
        if args.knowledge_root is not None:
            options['translation']['knowledge_root'] = args.knowledge_root
        if args.knowledge_manifest_sha256:
            options['translation']['knowledge_manifest_sha256'] = args.knowledge_manifest_sha256
    return options


# 身份串用于隔离翻译缓存
# 基础身份绑定知识manifest
# 增量审核另加资源指纹
# 各质量阶段追加版本段
# 同名窗口配置不可混用
# reviewed94在此校验载入
# convergence先构造词汇视图
# 复用现有安全知识来源
# 不读取质量评测答案
# partial保存服务创建参数
# 真正的模型由Worker懒建
# OCR和翻译工厂各自独立
# 语义分组开关传到窗口
# 窗口身份传到翻译会话
# 保存结果另保留原运行身份
# 当前配置不覆盖旧结果身份
# 参数与身份必须同步维护
def create_window(args: argparse.Namespace, **kwargs) -> MainWindow:
    """Shared by the real desktop entry point and product integration tests."""
    options = runtime_options(args)
    identity = 'legacy'
    reviewed_knowledge=None
    if args.knowledge_mode != 'off':
        from mechanical_drawing_translator.knowledge_v2_runtime import MANIFEST_SHA256, runtime_identity
        identity = runtime_identity(args.knowledge_mode, args.knowledge_manifest_sha256 or MANIFEST_SHA256)
        if args.enable_v2_quality_optimization:
            identity += ':quality.1'
        if args.enable_reviewed94:
            from mechanical_drawing_translator.reviewed94 import load_packaged
            reviewed_knowledge=load_packaged()
            identity+=':reviewed94.1:'+reviewed_knowledge.fingerprint
        if args.enable_integrated_quality:
            identity+=':integrated.1'
        if args.enable_professional_quality:
            identity+=':professional.1'
        if args.enable_reference_quality:
            identity+=':reference.1'
        if args.enable_convergence_quality:
            identity+=':convergence.2'
        if args.enable_argos_core:
            identity+=':argoscore.1'
        if getattr(args,'enable_final_optimization',False):
            identity+=':final.1'
        if getattr(args, 'product_core', False):
            identity+=':product.1'
        if getattr(args,'enable_semantic_edit',False):
            identity+=':semantic-edit.1'
    # 收敛词汇由运行包派生。
    # 基础Glossary保持独立只读来源。
    # 这里只准备轻量知识视图。
    # OCR和翻译模型仍由Worker拥有。
    convergence_glossary=None
    if args.enable_convergence_quality:
        from mechanical_drawing_translator.knowledge_v2_runtime import RuntimeKnowledge,default_runtime_root,MANIFEST_SHA256
        from mechanical_drawing_translator.translation_glossary import Glossary,default_glossary_path
        convergence_runtime=RuntimeKnowledge.load(args.knowledge_root or default_runtime_root(),
            args.knowledge_manifest_sha256 or MANIFEST_SHA256)
        convergence_glossary=convergence_runtime.glossary(Glossary.load(default_glossary_path()))
    return MainWindow(
        enable_runtime_preload=bool(args.enable_argos_core),
        translation_service_factory=partial(make_translation_service, **options['translation']),
        ocr_service_factory=partial(make_ocr_service, **options['ocr']),
        enable_semantic_grouping=options['enable_semantic_grouping'],
        enable_ocr_refinement=options['ocr']['enable_refinement'],
        enable_d8c_semantic=options['enable_d8c_semantic'], **kwargs,
        translation_runtime_identity=identity,
        reviewed94_knowledge=reviewed_knowledge,
        enable_integrated_quality=options.get('enable_integrated_quality',False),
        enable_professional_quality=args.enable_professional_quality,
        enable_convergence_quality=args.enable_convergence_quality,
        convergence_glossary=convergence_glossary,
    )


# 只消费程序认识的参数
# 其余参数交给Qt处理
# 高DPI策略先于QApplication
# 窗口尺寸使用Qt逻辑像素
# 字体按本机可用顺序选择
# 未安装字体不自动联网获取
# 样式表从包内相对位置读取
# 缺少资源应由实际启动暴露
# 日志只记录必要操作元数据
# 窗口信号在创建后附加日志
# 关闭信号记录进程结束
# 事件循环退出码传给调用者
def main() -> int:
    from mechanical_drawing_translator.production_logging import record_event, attach_window_logging
    args, qt_args = _argument_parser().parse_known_args(sys.argv[1:])
    record_event('startup', model_version='argos-en-zh-1.9', knowledge_version=args.knowledge_mode)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication([sys.argv[0], *qt_args])
    families = QFontDatabase.families()
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC",
                   "Noto Sans CJK SC", "Segoe UI"):
        if family in families:
            app.setFont(QFont(family, 10))
            break
    # 样式资源必须随源码或分发包存在。
    # 当前入口没有缺资源时的网络回退。
    # 字体选择与业务源坐标无关。
    # 不能据本机字体可用推断所有Win10机器。
    stylesheet = project_resource('mechanical_drawing_translator', 'styles.qss')
    app.setStyleSheet(stylesheet.read_text(encoding="utf-8"))
    window = create_window(args)
    attach_window_logging(window)
    app.aboutToQuit.connect(lambda: record_event('shutdown'))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

