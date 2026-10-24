"""资源占用控制。

转换时默认刻意保守：CPU 线程限制在少数几个、内存走低占用模式，
避免把用户正在玩的游戏顶掉。

优先级：显式设置 > 环境变量 > settings.json > 内置默认

环境变量：
    MP3NBS_THREADS     推理线程数（1-64）
    MP3NBS_LOW_MEMORY  1/0，低内存模式
"""

from __future__ import annotations

import json
import os

from .paths import work_dir

_SETTINGS_FILE = work_dir() / "settings.json"

_runtime: dict = {"threads": None, "low_memory": None}


def _cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count() or 4


def default_threads() -> int:
    """保守默认：一个 12 核机器只用 4 个线程，剩下的留给游戏。"""
    n = _cpu_count()
    if n <= 2:
        return 1
    if n <= 4:
        return 2
    return max(2, min(6, n // 3))


def load_settings() -> dict:
    try:
        data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(patch: dict) -> dict:
    cur = load_settings()
    cur.update({k: v for k, v in patch.items() if k in ("threads", "low_memory")})
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(
            json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    return cur


def _env_bool(name: str):
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on", "y"):
        return True
    if v in ("0", "false", "no", "off", "n"):
        return False
    return None


def threads() -> int:
    if _runtime["threads"]:
        return int(_runtime["threads"])
    env = os.environ.get("MP3NBS_THREADS", "").strip()
    if env.isdigit() and int(env) > 0:
        return min(64, int(env))
    saved = load_settings().get("threads")
    if isinstance(saved, int) and saved > 0:
        return min(64, saved)
    return default_threads()


def low_memory() -> bool:
    if _runtime["low_memory"] is not None:
        return bool(_runtime["low_memory"])
    env = _env_bool("MP3NBS_LOW_MEMORY")
    if env is not None:
        return env
    saved = load_settings().get("low_memory")
    if isinstance(saved, bool):
        return saved
    return True


def set_threads(n: int) -> None:
    _runtime["threads"] = max(1, min(64, int(n)))


def set_low_memory(flag: bool) -> None:
    _runtime["low_memory"] = bool(flag)


def apply_env_defaults() -> None:
    """把线程数写进各数学库的环境变量。

    必须在 numpy / onnxruntime 导入之前调用才最有效。
    """
    n = str(threads())
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, n)


def lower_process_priority() -> None:
    """把当前进程优先级降到「低于正常」。

    这样用户同时在玩游戏时，操作系统会优先调度游戏进程，
    转换只是被"插空"跑，不影响游戏帧率。
    """
    try:
        if os.name == "nt":
            import ctypes

            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            IDLE_PRIORITY_CLASS = 0x00000040
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            # 低于正常即可：完全 idle 反而会让转换慢到没边
            ctypes.windll.kernel32.SetPriorityClass(
                handle, BELOW_NORMAL_PRIORITY_CLASS
            )
        else:
            os.nice(5)
    except Exception:
        pass


def restore_process_priority() -> None:
    """恢复默认优先级。"""
    try:
        if os.name == "nt":
            import ctypes

            NORMAL_PRIORITY_CLASS = 0x00000020
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, NORMAL_PRIORITY_CLASS)
        else:
            os.nice(-5)
    except Exception:
        pass


def describe() -> dict:
    return {
        "threads": threads(),
        "low_memory": low_memory(),
        "cpu_count": _cpu_count(),
        "default_threads": default_threads(),
    }
