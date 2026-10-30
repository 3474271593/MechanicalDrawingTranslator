# mechanical_drawing_translator/resource_paths.py
"""源码与未来冻结目录共用的只读资源定位；不使用进程工作目录。"""
from pathlib import Path


def _inside(root: Path, parts: tuple[str, ...]) -> Path:
    # 资源声明只接受相对于受控根目录的路径；调用方不能借绝对路径切换盘符。
    # 显式用户模型路径仍由各服务的配置参数处理，不经过此默认资源入口。
    relative = Path(*parts)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('resource path must stay inside its declared root')
    # resolve消除工作目录、符号链接和冗余分隔符差异，再检查最终归属。
    # 不在定位时打开文件：服务保留自己的缺失/格式/哈希错误语义。
    root = root.resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('resource path escapes its declared root')
    return result


def package_resource(*parts: str) -> Path:
    # PyInstaller按spec将资源放在包目录的resources下；__file__随安装位置移动。
    # 不依赖固定用户名、开发工作区、sys.executable旁的旧dist或虚拟环境路径。
    return _inside(Path(__file__).resolve().parent / 'resources', parts)


def project_resource(*parts: str) -> Path:
    # data与顶层许可证在源码和未来_internal中都位于包目录的同级。
    # 保持同一相对目录合同，使直接运行与冻结运行无需两套业务分支。
    return _inside(Path(__file__).resolve().parent.parent, parts)
