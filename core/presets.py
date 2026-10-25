"""自动预设：根据音频特征挑一组转换参数，让懒人模式零配置也能出好结果。

分析维度（只看伴奏，不需要完整转录）：
  * 起音密度      —— 每秒有多少个音头，决定 onset / min_note 阈值
  * 节奏规整度    —— 音头间隔的离散程度，决定是否适合密集音符
  * 频谱重心      —— 高频占比高说明亮（适合长笛/铃），低说明厚（适合贝斯/大提琴）
  * 动态范围      —— 强弱对比大，需要放宽阈值保留弱音

输出是一组 ConvertOptions 字段的覆盖值 + 可读的说明。
"""

from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 512


def _stft_mag(mono: np.ndarray) -> np.ndarray:
    """幅度谱 (freq, frames)。"""
    win = np.hanning(N_FFT).astype(np.float32)
    pad = N_FFT // 2
    xp = np.pad(mono.astype(np.float32), (pad, pad), mode="reflect")
    n = 1 + max(0, (len(xp) - N_FFT) // HOP)
    if n <= 0:
        return np.zeros((N_FFT // 2 + 1, 1), dtype=np.float32)
    idx = np.arange(n)[:, None] * HOP + np.arange(N_FFT)[None, :]
    frames = xp[idx] * win
    return np.abs(np.fft.rfft(frames, axis=-1)).T.astype(np.float32)


def analyze(samples: np.ndarray, sr: int) -> dict:
    """分析音频，返回特征字典。samples 是 (channels, n) 或 (n,)。"""
    if samples.ndim > 1:
        mono = samples.mean(axis=0).astype(np.float32)
    else:
        mono = np.asarray(samples, dtype=np.float32).ravel()

    if mono.size < sr * 0.2:
        return {"duration": mono.size / max(sr, 1), "too_short": True}

    peak = float(np.abs(mono).max())
    if peak > 1e-9:
        mono = mono / peak

    mag = _stft_mag(mono)
    freqs = np.linspace(0, sr / 2, mag.shape[0])

    # --- 起音密度：短时能量的正向差值 ---
    energy = (mag ** 2).mean(axis=0)
    if energy.size > 1:
        diff = np.diff(energy)
        thr = max(float(np.abs(diff).mean()) * 1.5, 1e-8)
        onsets = int((diff > thr).sum())
    else:
        onsets = 0
    duration = mono.size / sr
    onset_rate = onsets / max(duration, 1e-6)

    # --- 频谱重心（能量加权） ---
    spec_mean = mag.mean(axis=1)
    total = float(spec_mean.sum()) + 1e-9
    centroid = float((spec_mean * freqs).sum() / total)

    # --- 高频占比（2kHz 以上）---
    hi = float(spec_mean[freqs >= 2000].sum() / total)
    lo = float(spec_mean[freqs < 250].sum() / total)

    # --- 动态范围：帧能量的 dB 差 ---
    e = energy + 1e-9
    db = 10 * np.log10(e)
    dr = float(np.percentile(db, 95) - np.percentile(db, 10))

    # --- 节奏规整度：能量相邻峰间隔的离散度 ---
    if energy.size > 8:
        idx = np.where(diff > max(float(np.abs(diff).mean()) * 2, 1e-8))[0]
        if idx.size >= 3:
            gaps = np.diff(idx).astype(np.float32)
            reg = float(gaps.std() / (gaps.mean() + 1e-6))
        else:
            reg = 1.0
    else:
        reg = 1.0

    return {
        "duration": duration,
        "too_short": False,
        "onset_rate": onset_rate,
        "centroid": centroid,
        "hi_ratio": hi,
        "lo_ratio": lo,
        "dynamic_db": dr,
        "rhythm_regularity": reg,
    }


def suggest_options(feat: dict) -> dict:
    """把特征映射成一组推荐参数（只返回需要覆盖的字段）。

    返回 dict 还会带 "_reason" 说明，供界面展示。
    """
    if feat.get("too_short"):
        return {
            "onset_threshold": 0.45,
            "frame_threshold": 0.25,
            "min_note_ms": 100.0,
            "instrument": "AUTO",
            "bass_instrument": "AUTO",
            "_reason": "音频较短，用宽松阈值尽量多识别",
        }

    rate = feat["onset_rate"]
    hi = feat["hi_ratio"]
    lo = feat["lo_ratio"]
    dr = feat["dynamic_db"]
    reg = feat["rhythm_regularity"]

    # --- 起音阈值：音头越密，阈值越高（避免糊成一片）---
    # 注意：音符盒想要「每 tick 都有音」，阈值不能太严。
    # 实测同一首歌：onset .62/frame .30/minlen 110 只有 330 音符、47% 的 tick 是多音；
    # 放宽到 .45/.20/80 能到 787 音符、77% 多音 tick。
    if rate > 8:
        onset, min_note = 0.45, 80.0
        density = "密集"
    elif rate > 4:
        onset, min_note = 0.42, 90.0
        density = "中等"
    elif rate > 1.5:
        onset, min_note = 0.40, 110.0
        density = "偏慢"
    else:
        onset, min_note = 0.36, 130.0
        density = "舒缓"

    # --- 帧阈值：这是保留弱音的关键，越低留下的音越多 ---
    if dr > 45:
        frame = 0.15
    elif dr > 32:
        frame = 0.18
    else:
        frame = 0.20

    # --- 乐器倾向（仍交给 AUTO 细选，这里只给个偏向）---
    # 高频多说明亮，适合长笛/铃；低频厚适合贝斯类
    if hi > 0.35 and lo < 0.15:
        hint = "明亮，倾向长笛/铃"
    elif lo > 0.32:
        hint = "低频厚，倾向大提琴/贝斯"
    else:
        hint = "音色均衡，倾向竖琴/钢琴"

    # --- 中继器粒度：节奏越规整，越能压缩 ---
    coarse = 2 if (reg < 0.35 and rate > 4) else 1 if reg < 0.6 else 0

    reason = (
        f"节奏{density}（{rate:.1f} 音/秒）· 动态 {dr:.0f} dB · "
        f"{hint} · 中继器粒度 {coarse}"
    )

    return {
        "onset_threshold": round(onset, 2),
        "frame_threshold": round(frame, 2),
        "min_note_ms": round(min_note, 1),
        "coarse": coarse,
        "instrument": "AUTO",
        "bass_instrument": "AUTO",
        "_reason": reason,
    }
