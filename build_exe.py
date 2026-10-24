"""PyInstaller 打包脚本。

用法：
    python build_exe.py              # 打目录版（默认，启动 0.4 秒）
    python build_exe.py --zip        # 目录版 + 便携版 ZIP
    python build_exe.py --onefile    # 单文件 exe（⚠️ 启动要 2-3 分钟，不推荐）

产物：
    dist/MP3NBS转换器/             —— 目录版（安装包与便携版都用它）
    dist/MP3NBS转换器_便携版.zip   —— 免安装解压即用
    dist/MP3NBS转换器_安装包.exe   —— 由 installer.iss 生成

⚠️ 关于单文件版：
    onefile 每次启动都要把 ~520 MB 解包到临时目录，实测冷启动 180 秒、
    热启动 149 秒（还要被杀毒扫描一遍）；而目录版只要 0.38 秒。
    所以默认不再打单文件版，需要时显式加 --onefile，并建议用便携 ZIP 代替。

说明：
  * 目录版把 models/ 拷到 exe 同级（不打进包，避免 550 MB 存两份），
    用户可直接替换模型文件。
  * web/ 静态文件打进包内，从 _MEIPASS 读取。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent
MODELS = PROJ / "models"
WEB = PROJ / "web"
DIST = PROJ / "dist"
BUILD = PROJ / "build"

APP_NAME = "MP3NBS转换器"
ENTRY = PROJ / "run.py"
ICON = PROJ / "assets" / "app.ico"

# 需要排除的大依赖（用不到，别拖进包里）
EXCLUDES = [
    "matplotlib",
    "pandas",
    "IPython",
    "jupyter",
    "notebook",
    "tkinter",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "pytest",
    "sphinx",
    "setuptools",
    "pip",
    "wheel",
    "sqlite3",
    "librosa.display",
    "numba",
    "llvmlite",
]

# 需要额外收集的 hidden imports
HIDDEN = [
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
    "scipy.signal",
    "scipy.special",
    "scipy.fft",
    "scipy.linalg",
    "librosa",
    "librosa.core",
    "librosa.util",
    "soundfile",
    "mido",
    "pynbs",
    "fastapi",
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.lifespan",
    "multipart",
    "anyio",
    "starlette",
    "starlette.responses",
    "starlette.staticfiles",
]


def build(onefile: bool = True, clean: bool = True) -> int:
    if clean:
        for d in (BUILD, DIST):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    sep = ";" if os.name == "nt" else ":"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        APP_NAME,
        "--console",
        "--onedir" if not onefile else "--onefile",
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD),
        "--specpath",
        str(BUILD),
    ]

    # 数据文件
    #   web/ 必须打进包里（打包后从 _MEIPASS 读）
    #   models/ 只有单文件版才打进包；目录版直接拷到 exe 同级，
    #   否则 550 MB 模型会被存两份。
    data_srcs = [WEB] if not onefile else [MODELS, WEB]
    for src in data_srcs:
        if not src.exists():
            print(f"[警告] 缺少 {src}，跳过")
            continue
        cmd += ["--add-data", f"{src}{sep}{src.name}"]

    for hidden in HIDDEN:
        cmd += ["--hidden-import", hidden]
    for exc in EXCLUDES:
        cmd += ["--exclude-module", exc]

    if ICON.exists():
        cmd += ["--icon", str(ICON)]

    cmd += ["--collect-all", "onnxruntime"]
    cmd += ["--collect-all", "pynbs"]

    cmd.append(str(ENTRY))

    print("执行：")
    print("  " + " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=str(PROJ))
    if result.returncode != 0:
        print("\n[错误] PyInstaller 失败")
        return result.returncode

    # 目录版：把 models/ 拷到 exe 同级，方便用户替换（web 已在 _internal 里，不重复拷）
    if not onefile:
        out_dir = DIST / APP_NAME
        for name in ("models",):
            src = PROJ / name
            dst = out_dir / name
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

    exe = DIST / (APP_NAME + (".exe" if os.name == "nt" else ""))
    if onefile:
        exe = DIST / APP_NAME
    print()
    if exe.exists():
        size = exe.stat().st_size / (1024 * 1024)
        print(f"[完成] {exe}  ({size:.1f} MB)")
    else:
        print(f"[完成] 产物在 {DIST}")
    return 0


def make_zip() -> Path | None:
    """把目录版打成便携 ZIP（解压即用，启动仍然 0.4 秒）。"""
    src = DIST / APP_NAME
    if not src.is_dir():
        return None
    out = DIST / f"{APP_NAME}_便携版.zip"
    print(f"\n打包便携 ZIP：{out}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, Path(APP_NAME) / p.relative_to(src))
    size = out.stat().st_size / (1024 * 1024)
    print(f"[完成] {out}  ({size:.1f} MB)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="打包成 exe")
    p.add_argument("--onedir", action="store_true", help="打目录版（默认行为，可省略）")
    p.add_argument("--onefile", action="store_true",
                   help="打单文件 exe（⚠️ 启动要 2-3 分钟，一般不建议）")
    p.add_argument("--zip", action="store_true", help="目录版 + 便携版 ZIP")
    p.add_argument("--both", action="store_true", help="目录版 + 单文件版")
    p.add_argument("--no-clean", action="store_true", help="不清理旧的 build/dist")
    args = p.parse_args()

    clean = not args.no_clean

    if args.onefile:
        return build(onefile=True, clean=clean)
    if args.both:
        rc = build(onefile=False, clean=clean)
        if rc != 0:
            return rc
        return build(onefile=True, clean=False)

    rc = build(onefile=False, clean=clean)
    if rc != 0:
        return rc
    if args.zip:
        make_zip()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
