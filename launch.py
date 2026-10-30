# launch.py
"""Project-owner daily entry; the customer release entry main.py keeps its stable default."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


# 入口只组合既有参数
# 模型由服务工厂创建
# daily含stage3候选
# stable回稳定配置
# v1仍继承daily配置
# v2另加知识运行模式
# overlay复用v2候选
# reviewed94增加审核增量
# integrated叠加质量开关
# professional启用专业门禁
# reference复用受控知识
# convergence追加收敛逻辑
# argos-core追加后端约束
# product再加正式核心标记
# product不继承final分支
# 因此不启用ORT图缓存
# 也不启用semantic-edit
# 各入口仍共享一个主窗口
# 参数顺序体现父配置关系
# 未知模式明确报错
def entry_arguments(mode: str) -> list[str]:
    """Reuse existing product profiles, without another OCR implementation or mutable settings."""
    if mode == 'daily':
        return ['--runtime-profile', 'd7d8-reviewed-candidate', '--enable-ocr-stage3-trial']
    if mode == 'v2-quality':
        return [*entry_arguments('v2-candidate'), '--enable-v2-quality-optimization']
    if mode == 'overlay':
        return entry_arguments('v2-candidate')
    if mode == 'reviewed94':
        return [*entry_arguments('overlay'),'--enable-reviewed94']
    if mode == 'integrated':
        return [*entry_arguments('reviewed94'),'--enable-v2-quality-optimization','--enable-integrated-quality']
    if mode == 'professional':
        return [*entry_arguments('integrated'),'--enable-professional-quality']
    if mode == 'convergence':
        return [*entry_arguments('reference'),'--enable-convergence-quality']
    if mode == 'argos-core':
        return [*entry_arguments('convergence'),'--enable-argos-core']
    if mode == 'product':
        return [*entry_arguments('argos-core'), '--product-core']
    if mode == 'final-optimization':
        return [*entry_arguments('argos-core'),'--enable-final-optimization']
    if mode == 'semantic-edit':
        return [*entry_arguments('final-optimization'),'--enable-semantic-edit']
    if mode == 'reference':
        return [*entry_arguments('professional'),'--enable-reference-quality']
    if mode in {'v2-shadow', 'v2-candidate', 'v1'}:
        values = entry_arguments('daily')
        return values if mode == 'v1' else [*values, '--knowledge-mode', mode.removeprefix('v2-')]
    if mode == 'stage2':
        return ['--runtime-profile', 'd7d8-reviewed-candidate', '--enable-ocr-stage2-trial']
    if mode == 'stable':
        return ['--runtime-profile', 'stable']
    raise ValueError(f'Unknown launch mode: {mode}')


# 此处解析的是启动预设
# 不是main的全部业务参数
# 直接运行默认daily
# 工作区启动器可指定product
# 选择预设不修改磁盘配置
def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('product', 'daily', 'stable', 'stage2', 'v1', 'v2-shadow', 'v2-candidate', 'v2-quality','overlay','reviewed94','integrated','professional','reference','convergence','argos-core','final-optimization','semantic-edit'), default='daily',
                        help='daily为负责人日常统一候选；stable回原默认预设；stage2回上一轮候选')
    return parser.parse_args(argv)


# main读取进程参数
# 临时替换为预设参数
# finally恢复原参数对象
# 嵌入调用也不会遗留参数
# 不在这里创建第二套UI
# 退出码沿用桌面事件循环
def run(mode: str = 'daily') -> int:
    arguments = entry_arguments(mode)
    import main as desktop

    previous = sys.argv
    try:
        sys.argv = [str(Path(__file__).with_name('main.py')), *arguments]
        return desktop.main()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(run(parse_arguments().mode))
