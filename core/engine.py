"""可选组件：nbs-fractal 排布引擎的下载与调用。

这个二进制不在安装包里（AGPL-3.0 项目的产物不宜直接打包分发），
改成运行时按需下载到 ``tools/`` 目录：
  * 存在 → 用它生成与 u-u.life/nbs 完全一致的 litematic
  * 不存在 → pipeline 抛 LayoutEngineMissing，由界面询问用户
             是「下载引擎」还是「接受效果较差的内置排布」

下载地址指向本项目 Release 的镜像产物，来源与协议在 README 里注明。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
import tempfile
import urllib.request
from pathlib import Path

#: 引擎在 Goal 里的文件名
ENGINE_NAME = "nbs-fractal.exe"

#: 下载源：本项目 Release 里镜像的 nbs-fractal 构建产物。
#: 换版本时改这里即可（支持多个镜像，按顺序尝试）。
ENGINE_URLS = [
    "https://github.com/szkele1145/mp3-nbs-converter/releases/download/tools/nbs-fractal.exe",
]

#: 校验值（首次编译出来后填入，留空则跳过校验）
ENGINE_SHA256 = ""

#: 下载超时（秒）
DOWNLOAD_TIMEOUT = 180


def tools_dir() -> Path:
    """引擎存放目录：优先 exe/项目同级 tools/，不可写时退到用户目录。"""
    from .paths import app_dir

    d = app_dir() / "tools"
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".writable"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return d
    except OSError:
        # 只读安装位置 → 退到用户目录
        fallback = Path.home() / ".mp3-nbs-converter" / "tools"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def engine_path() -> Path:
    """引擎的期望路径。"""
    return tools_dir() / ENGINE_NAME


def find_engine() -> str | None:
    """找一个可用的引擎：先看 tools/，再看 PATH。"""
    p = engine_path()
    if p.is_file():
        return str(p)
    found = shutil.which("nbs-fractal")
    return found if found else None


def is_available() -> bool:
    return find_engine() is not None


def _verify(path: Path) -> bool:
    """校验下载的文件。留空校验值时只做基本检查。"""
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    if not ENGINE_SHA256:
        return True
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h.lower() == ENGINE_SHA256.lower()


def _ssl_context() -> ssl.SSLContext:
    """建 SSL 上下文。

    Windows 上证书吊销服务器常常连不上（CRYPT_E_REVOCATION_OFFLINE），
    导致所有 HTTPS 请求直接失败。这里关掉吊销检查，
    仍保留证书链校验（MITM 依然会被拦下）。
    """
    ctx = ssl.create_default_context()
    try:
        ctx.verify_flags &= ~ssl.VERIFY_CRL_CHECK_LEAF
        ctx.verify_flags &= ~ssl.VERIFY_CRL_CHECK_CHAIN
    except AttributeError:
        pass
    # 部分环境没有 CRL 支持，直接放开吊销相关开关
    try:
        ctx.verify_flags = ssl.VERIFY_X509_PARTIAL_CHAIN | ssl.VERIFY_X509_STRICT  # type: ignore[assignment]
    except (AttributeError, TypeError):
        pass
    return ctx


def download(progress=None) -> tuple[bool, str]:
    """下载引擎到 tools/。

    progress: 可调用对象 progress(ratio: float) —— 目前只有起止两个点
    返回 (成功与否, 说明文字)。
    """
    dest = engine_path()
    last_err = ""
    ctx = _ssl_context()

    for url in ENGINE_URLS:
        tmp_path = None
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "mp3-nbs-converter", "Accept": "*/*"},
            )
            fd, tmp_name = tempfile.mkstemp(suffix=".exe", dir=str(tools_dir()))
            os.close(fd)
            tmp_path = Path(tmp_name)

            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT, context=ctx) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp_path, "wb") as fh:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None and total > 0:
                            try:
                                progress(min(0.99, done / total))
                            except Exception:  # noqa: BLE001
                                pass

            if not _verify(tmp_path):
                last_err = "下载的文件校验失败"
                tmp_path.unlink(missing_ok=True)
                continue

            tmp_path.replace(dest)
            if progress is not None:
                try:
                    progress(1.0)
                except Exception:  # noqa: BLE001
                    pass
            return True, f"排布引擎已就绪（{dest.stat().st_size / 1024 / 1024:.1f} MB）"

        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    return False, f"下载失败：{last_err or '没有可用的下载源'}"


def run_compact(
    nbs_path: str,
    out_path: str,
    wrap: int = 16,
    coarse: int = 0,
    gap: int = 0,
) -> tuple[bool, str]:
    """用 nbs-fractal compact 生成 litematic。

    参数语义与 nbs-fractal CLI 一致：
      wrap   每行最大列数（0 = 不换行）
      coarse 中继器粒度 1-4（0 = 不限）
      gap    音轨间距（0 = 咬合）
    返回 (成功与否, 说明)。
    """
    exe = find_engine()
    if not exe:
        return False, "排布引擎未安装"

    cmd = [
        exe, "compact", str(nbs_path), str(out_path),
        "--wrap", str(max(0, int(wrap))),
        "--coarse", str(max(0, min(4, int(coarse)))),
        "--gap", str(max(0, int(gap))),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
    except subprocess.TimeoutExpired:
        return False, "排布引擎超时"
    except Exception as exc:  # noqa: BLE001
        return False, f"排布引擎启动失败：{exc}"

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()[-300:]
        return False, f"排布引擎返回错误：{msg}"
    if not Path(out_path).is_file():
        return False, "排布引擎未产出文件"
    return True, Path(out_path).name
