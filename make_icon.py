"""生成应用图标 assets/app.ico（纯 numpy 绘制，无需 Pillow 依赖以外的工具）。

画一个「音符盒 + 音符」的风格化图标：
  圆角深色底 + 三根音符柱（像音符盒按键）+ 一个金色音符符号。
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "assets"
SIZES = (16, 32, 48, 64, 128, 256)


def draw(size: int) -> np.ndarray:
    """返回 (size, size, 4) 的 RGBA 图像。"""
    s = size
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float64)
    px = (xx + 0.5) / s
    py = (yy + 0.5) / s

    img = np.zeros((s, s, 4), dtype=np.float32)
    img[..., 3] = 0.0

    # ---- 圆角矩形底 ----
    r = 0.18
    m = 0.04
    cx = np.clip(px, m + r, 1 - m - r)
    cy = np.clip(py, m + r, 1 - m - r)
    d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    inside = (px > m) & (px < 1 - m) & (py > m) & (py < 1 - m) & (d <= r)
    # 竖直渐变：深蓝 → 紫
    top = np.array([0.13, 0.15, 0.28], dtype=np.float32)
    bot = np.array([0.22, 0.13, 0.36], dtype=np.float32)
    grad = top[None, None, :] * (1 - py[..., None]) + bot[None, None, :] * py[..., None]
    img[inside, :3] = grad[inside]
    img[inside, 3] = 1.0

    def disc(cx0, cy0, r0, color, alpha=1.0):
        d0 = np.sqrt((px - cx0) ** 2 + (py - cy0) ** 2)
        m0 = d0 <= r0
        msoft = (d0 > r0) & (d0 < r0 + 0.02)
        if m0.any():
            img[m0, :3] = img[m0, :3] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
            img[m0, 3] = 1.0
        if msoft.any():
            w = 1.0 - (d0[msoft] - r0) / 0.02
            img[msoft, :3] = img[msoft, :3] * (1 - w[..., None] * alpha) + \
                np.array(color, dtype=np.float32) * (w[..., None] * alpha)
            img[msoft, 3] = 1.0

    def rect(x0, y0, x1, y1, color, alpha=1.0):
        m0 = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        if m0.any():
            img[m0, :3] = img[m0, :3] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
            img[m0, 3] = 1.0

    gold = (0.98, 0.78, 0.32)
    light = (0.86, 0.90, 1.0)

    # ---- 音符盒按键：三根不同高度的柱 ----
    bars = [(0.24, 0.62), (0.44, 0.42), (0.64, 0.54)]
    for bx, top_y in bars:
        rect(bx - 0.055, top_y, bx + 0.055, 0.80, light, 0.92)
        # 顶面高光
        rect(bx - 0.055, top_y, bx + 0.055, top_y + 0.035, gold, 0.95)

    # ---- 音符符头（两个圆）+ 符干 ----
    disc(0.36, 0.30, 0.115, gold)
    disc(0.60, 0.24, 0.100, gold)
    rect(0.455, 0.26, 0.495, 0.42, gold, 0.95)
    rect(0.680, 0.20, 0.720, 0.38, gold, 0.95)
    # 符尾横杠
    rect(0.455, 0.235, 0.720, 0.275, gold, 0.95)

    # 底座
    rect(0.18, 0.82, 0.82, 0.87, gold, 0.75)

    return img


def save_ico(path: Path) -> None:
    """写一个多尺寸 ICO（内嵌 PNG）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    pngs = []
    for s in SIZES:
        # draw() 返回 0..1 的 float，必须 ×255 再转 uint8，
        # 否则 alpha 只落在 0/1 两个值上，图标看起来是空白的。
        img = (np.clip(draw(s), 0.0, 1.0) * 255).round().astype(np.uint8)
        pngs.append(encode_png(img))

    n = len(pngs)
    header = struct.pack("<HHH", 0, 1, n)
    offset = 6 + 16 * n
    entries = []
    for s, data in zip(SIZES, pngs):
        entries.append(struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32,
                                  len(data), offset))
        offset += len(data)
    with path.open("wb") as f:
        f.write(header)
        for e in entries:
            f.write(e)
        for data in pngs:
            f.write(data)


def encode_png(rgba: np.ndarray) -> bytes:
    """把 (H, W, 4) uint8 数组编成 PNG 字节（只用 zlib，不依赖 Pillow）。"""
    import zlib

    h, w, _ = rgba.shape
    raw = np.zeros((h, w * 4 + 1), dtype=np.uint8)
    raw[:, 1:] = rgba.reshape(h, w * 4)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    out = b"\x89PNG\r\n\x1a\n"
    out += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    out += chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
    out += chunk(b"IEND", b"")
    return out


def main() -> None:
    p = OUT_DIR / "app.ico"
    save_ico(p)
    print(f"已生成 {p}  ({p.stat().st_size:,} 字节)")
    # 另外存一张 PNG 预览，方便查看
    png = OUT_DIR / "app.png"
    png.write_bytes(encode_png((draw(256) * 255).astype(np.uint8)))
    print(f"已生成 {png}")


if __name__ == "__main__":
    main()
