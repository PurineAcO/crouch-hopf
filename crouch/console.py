"""控制台输出美化工具: ANSI 颜色 + 分节标题 + 成功/警告/错误分级."""

import sys

# ANSI 转义码
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _enable_ansi() -> None:
    """Windows 下启用 VT 处理(使 ANSI 颜色生效), 其他平台无需处理."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)                       # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING


_enable_ansi()


def ok(msg: str) -> None:
    """成功信息(绿色, 带 ✔ 前缀)."""
    print(f"{GREEN}✔ {msg}{RESET}")


def ok_raw(msg: str) -> None:
    """成功信息(绿色, 无前缀)."""
    print(f"{GREEN}{msg}{RESET}")


def info(msg: str) -> None:
    """过程信息(青色)."""
    print(f"{CYAN}{msg}{RESET}")


def warn(msg: str) -> None:
    """警告信息(橙色, 带 ⚠ 前缀)."""
    print(f"{YELLOW}⚠ {msg}{RESET}")


def warn_raw(msg: str) -> None:
    """警告信息(橙色, 无前缀)."""
    print(f"{YELLOW}{msg}{RESET}")


def err(msg: str) -> None:
    """错误信息(红色, 带 ✖ 前缀)."""
    print(f"{RED}✖ {msg}{RESET}")


def section(title: str) -> None:
    """分节大标题(青色加粗)."""
    width = max(4, 66 - len(title))
    print()
    print(f"{BOLD}{CYAN}═══ {title} ═{'═' * width}{RESET}")
