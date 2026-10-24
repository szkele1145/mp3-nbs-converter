"""路径解析：同时兼容「源码运行」与「PyInstaller 打包后运行」。

打包后（frozen）：
    可写目录  → exe 同级目录（便携模式）或 %LOCALAPPDATA%\<应用名>\work
    只读资源  → sys._MEIPASS（web 静态文件、models 模型）

源码运行：
    项目根 = 本文件的上级目录
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "MP3NBSConverter"

#: 是否运行在 PyInstaller 打包环境
FROZEN = bool(getattr(sys, "frozen", False))

#: PyInstaller 解包出来的只读资源根目录（打包后才有）
if FROZEN:
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    """应用根目录。

    打包后 = exe 所在目录（便携模式，用户可直接替换 models/ 里的模型）。
    """
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return BUNDLE_DIR


def resource_dir() -> Path:
    """只读资源目录（web 静态文件）。

    打包后 web/ 会被塞进 _MEIPASS；源码运行则是项目下的 web/。
    """
    if FROZEN:
        cand = BUNDLE_DIR / "web"
        if cand.exists():
            return cand
        # 单文件模式：静态文件可能被打进了归档，这里做一次解压
        return BUNDLE_DIR
    return BUNDLE_DIR / "web"


def models_dir() -> Path:
    """模型目录。

    优先找「可写」位置（便携模式：exe 同级 models/），
    找不到再回落到打包内的只读副本。
    """
    local = app_dir() / "models"
    if local.exists() and any(local.glob("*.onnx")):
        return local
    bundled = BUNDLE_DIR / "models"
    if bundled.exists():
        return bundled
    return local


def work_dir() -> Path:
    """可写工作目录（上传、输出、缓存）。"""
    if FROZEN:
        d = app_dir() / "work"
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".writable"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return d
        except OSError:
            # 只读位置（装到 Program Files 之类）→ 退回用户目录
            d = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "work"
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = BUNDLE_DIR / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def upload_dir() -> Path:
    d = work_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir() -> Path:
    d = work_dir() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """分离结果缓存目录（调音高参数时免去重复去人声）。"""
    d = work_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
