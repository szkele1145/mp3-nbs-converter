"""节拍检测与网格量化：把识别出的零散音符吸附到节拍网格上。

从音频转录出来的音符时间是连续的（0.337s、0.348s…），
而 NBS 每 0.1 秒只有一个 tick，不量化就会：
  * 本该同一拍的两个音被拆到相邻 tick
  * 每个 tick 平均不到 2 个音，投影稀疏

做法：
  1. 估 BPM（用自相关找拍点周期）
  2. 求首个拍点相位
  3. 把所有音符起音时刻吸附到最近的网格点，同音高的合并
"""

from __future__ import annotations

import numpy as np


def _onset_envelope(samples: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, float]:
    """简单的能量起音包络。返回 (envelope, 每帧秒数)。"""
    n_fft = 1024
    win = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(np.asarray(samples, dtype=np.float32), (pad, pad), mode="reflect")
    n_frames = 1 + max(0, (len(xp) - n_fft) // hop)
    if n_frames <= 1:
        return np.zeros(1, dtype=np.float32), hop / sr

    idx = np.arange(n_frames)[:, None] * hop + np.arange(n_fft)[None, :]
    frames = xp[idx] * win
    mag = np.abs(np.fft.rfft(frames, axis=-1))
    energy = (mag ** 2).mean(axis=1)

    # 正向差值 = 起音强度
    diff = np.diff(energy, prepend=energy[0])
    diff[diff < 0] = 0.0
    return diff.astype(np.float32), hop / sr


def estimate_bpm(samples: np.ndarray, sr: int, min_bpm: float = 60.0,
                 max_bpm: float = 200.0) -> float:
    """用自相关估 BPM。失败返回 0。"""
    env, frame_sec = _onset_envelope(samples, sr)
    if len(env) < 20:
        return 0.0

    # 去均值
    x = env - env.mean()
    if not np.any(x):
        return 0.0

    # 自相关（只算我们关心的 lag 范围）
    fps = 1.0 / frame_sec
    min_lag = int(max(1, fps * 60.0 / max_bpm))
    max_lag = int(fps * 60.0 / min_bpm)
    max_lag = min(max_lag, len(x) // 2)
    if max_lag <= min_lag:
        return 0.0

    corr = np.correlate(x, x, mode="full")[len(x) - 1:]
    seg = corr[min_lag:max_lag + 1]
    if len(seg) == 0 or seg.max() <= 0:
        return 0.0

    best_lag = int(np.argmax(seg)) + min_lag
    bpm = 60.0 * fps / best_lag
    # 倍频折回常用区间
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 190.0:
        bpm /= 2.0
    return float(bpm)


def quantize_notes(
    notes: list,
    bpm: float = 0.0,
    ticks_per_sec: float = 10.0,
    subdivisions: int = 1,
    tolerance_ticks: float = 0.6,
) -> list:
    """把音符起音吸附到网格，返回新的音符列表。

    bpm      0 = 不按 BPM，直接吸附到 ticks_per_sec 网格
    subdivisions  八分/十六分音符细分（1 = 每 tick 一格）
    tolerance_ticks  超过这个距离就不吸附（避免把自由拍压平）
    """
    if not notes:
        return notes

    # 网格步长（秒）
    if bpm and bpm > 0:
        # 一个四分音符 = 60/bpm 秒；再细分
        beat_sec = 60.0 / bpm
        step = beat_sec / subdivisions
        # 与 NBS tick 取整的最小公倍数式对齐：让网格尽量贴近 0.1s
        step = max(step, 1.0 / ticks_per_sec)
    else:
        step = 1.0 / ticks_per_sec

    out = []
    for n in notes:
        start = float(n.start)
        q = round(start / step) * step
        # 离网格太远就不动它（保留自由节奏）
        if abs(start - q) > tolerance_ticks * (1.0 / ticks_per_sec):
            q = start
        try:
            n2 = type(n)(
                start=q,
                duration=float(n.duration),
                midi=int(n.midi),
                velocity=float(n.velocity),
            )
        except TypeError:
            # 数据类签名不同就原地改
            n.start = q
            n2 = n
        out.append(n2)

    # 合并：同一时刻同一音高只留一个
    merged: dict[tuple[int, int], object] = {}
    for n in out:
        key = (int(round(float(n.start) * ticks_per_sec)), int(n.midi))
        prev = merged.get(key)
        if prev is None or float(n.duration) > float(prev.duration):
            merged[key] = n
    return sorted(merged.values(), key=lambda n: (n.start, n.midi))
