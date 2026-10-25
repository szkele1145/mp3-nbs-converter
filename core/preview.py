"""把 .nbs 合成成可播放的 WAV，用于试听。

不依赖外部采样：用合成音色近似 Minecraft 音符盒。
每个乐器给一个音色特征（谐波构成 + 衰减），听起来能区分鼓、贝斯、钢琴、长笛。

用法：
    render_nbs_to_wav("song.nbs", "preview.wav")
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

#: NBS 乐器名 → 音色参数
#: harmonics = 各谐波相对强度；decay = 衰减速度（越大越短促）；noise = 噪声成分
TIMBRES: dict[str, dict] = {
    "HARP":        {"harmonics": [1.0, 0.45, 0.22, 0.10], "decay": 14.0, "noise": 0.0},
    "PIANO":       {"harmonics": [1.0, 0.55, 0.30, 0.16, 0.08], "decay": 12.0, "noise": 0.0},
    "BASS":        {"harmonics": [1.0, 0.6, 0.15], "decay": 9.0,  "noise": 0.0},
    "DOUBLE_BASS": {"harmonics": [1.0, 0.5, 0.12], "decay": 8.0,  "noise": 0.0},
    "FLUTE":       {"harmonics": [1.0, 0.18, 0.06], "decay": 7.0,  "noise": 0.02},
    "BELL":        {"harmonics": [1.0, 0.0, 0.55, 0.0, 0.28], "decay": 9.0,  "noise": 0.0},
    "CHIME":       {"harmonics": [1.0, 0.0, 0.4, 0.0, 0.2], "decay": 10.0, "noise": 0.0},
    "PLING":       {"harmonics": [1.0, 0.3, 0.15], "decay": 12.0, "noise": 0.0},
    "GUITAR":      {"harmonics": [1.0, 0.5, 0.28, 0.12], "decay": 13.0, "noise": 0.01},
    "BANJO":       {"harmonics": [1.0, 0.45, 0.3], "decay": 16.0, "noise": 0.0},
    "XYLOPHONE":   {"harmonics": [1.0, 0.0, 0.3], "decay": 20.0,  "noise": 0.0},
    "IRON_XYLOPHONE": {"harmonics": [1.0, 0.0, 0.45, 0.0, 0.2], "decay": 18.0, "noise": 0.0},
    "BASS_DRUM":   {"harmonics": [1.0, 0.25], "decay": 22.0, "noise": 0.04, "pitch_drop": True},
    "SNARE_DRUM":  {"harmonics": [1.0, 0.4], "decay": 28.0,  "noise": 0.55},
    "CLICK":       {"harmonics": [1.0], "decay": 45.0,       "noise": 0.25},
    "COW_BELL":    {"harmonics": [1.0, 0.6, 0.3], "decay": 15.0, "noise": 0.0},
    "TRUMPET":     {"harmonics": [1.0, 0.7, 0.5, 0.3], "decay": 8.0, "noise": 0.0},
    "BIT":         {"harmonics": [1.0, 0.0, 0.5], "decay": 15.0, "noise": 0.0},
    "DIDGERIDOO":  {"harmonics": [1.0, 0.7, 0.4], "decay": 7.0, "noise": 0.03},
}

DEFAULT_TIMBRE = {"harmonics": [1.0, 0.4, 0.2], "decay": 4.0, "noise": 0.0}


def _midi_of_key(key: int) -> int:
    """NBS key → MIDI 音高（NBS key 45 = F#4 = MIDI 66，按 OpenNBS 惯例）。"""
    return int(key) + 21


def _synth_note(freq: float, dur: float, sr: int, timbre: dict, velocity: float) -> np.ndarray:
    """合成单个音。"""
    n = max(1, int(dur * sr))
    t = np.arange(n, dtype=np.float32) / sr

    wave = np.zeros(n, dtype=np.float32)
    for i, amp in enumerate(timbre.get("harmonics", [1.0])):
        if amp <= 0:
            continue
        f = freq * (i + 1)
        if f > sr / 2:
            break
        wave += amp * np.sin(2 * np.pi * f * t).astype(np.float32)

    # 打击乐的音高快速下滑，听感更像鼓
    if timbre.get("pitch_drop"):
        f0 = freq
        f1 = freq * 0.35
        phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * max(dur, 1e-3)))
        wave = 0.7 * np.sin(phase).astype(np.float32) + 0.3 * wave

    noise_amt = float(timbre.get("noise", 0.0))
    if noise_amt > 0:
        rng = np.random.default_rng(int(freq) & 0xFFFF)
        wave += noise_amt * rng.standard_normal(n).astype(np.float32)

    # 包络：快起音 + 指数衰减
    attack = min(int(0.004 * sr), max(1, n // 8))
    env = np.exp(-timbre.get("decay", 4.0) * t).astype(np.float32)
    env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    wave *= env

    peak = float(np.abs(wave).max())
    if peak > 1e-9:
        wave = wave / peak * float(velocity)
    return wave


def render_nbs_to_wav(
    nbs_path: str,
    wav_path: str,
    sr: int = 22050,
    max_seconds: float = 0.0,
    gain: float = 0.7,
) -> tuple[bool, str]:
    """把 .nbs 渲染成 WAV，返回 (成功, 说明)。"""
    try:
        import pynbs

        song = pynbs.read(nbs_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"读取 nbs 失败：{exc}"

    try:
        notes = list(song.notes)
    except Exception as exc:  # noqa: BLE001
        return False, f"解析音符失败：{exc}"

    if not notes:
        return False, "乐谱里没有音符"

    tempo = float(getattr(song.header, "tempo", 10.0) or 10.0)
    if tempo <= 0:
        tempo = 10.0
    tick_sec = 1.0 / tempo

    # 乐器表：nbs 里的 instrument 是索引
    inst_names: list[str] = []
    try:
        for ins in song.instruments:
            inst_names.append(str(getattr(ins, "name", "") or "").upper())
    except Exception:  # noqa: BLE001
        pass

    max_tick = max(int(n.tick) for n in notes)
    total_sec = (max_tick + 1) * tick_sec
    if max_seconds and max_seconds > 0:
        total_sec = min(total_sec, max_seconds)
        notes = [n for n in notes if int(n.tick) * tick_sec < max_seconds]

    n_samples = max(1, int(total_sec * sr))
    mix = np.zeros(n_samples, dtype=np.float32)

    # 按 tick 排序，算出每个音到下一个音的距离，用来自适应缩短时长
    sorted_notes = sorted(notes, key=lambda n: int(n.tick))
    tick_gaps: dict[int, float] = {}
    for i, n in enumerate(sorted_notes):
        cur = int(n.tick)
        nxt = int(sorted_notes[i + 1].tick) if i + 1 < len(sorted_notes) else cur + 1
        tick_gaps.setdefault(cur, max(1, nxt - cur) * tick_sec)

    for note in notes:
        start = int(note.tick) * tick_sec
        idx = int(start * sr)
        if idx >= n_samples:
            continue

        key = int(note.key)
        midi = _midi_of_key(key)
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))

        inst_idx = int(getattr(note, "instrument", 0) or 0)
        name = inst_names[inst_idx] if 0 <= inst_idx < len(inst_names) else "HARP"
        timbre = TIMBRES.get(name, DEFAULT_TIMBRE)

        # 音符盒本身是短促的「叮」，不是长音。
        # 之前固定 0.35s，而平均 0.1s 就有一个音，三四个音尾叠在一起就糊了
        # —— 这就是「前两秒清楚、后面听不懂」的原因。
        gap = tick_gaps.get(int(note.tick), tick_sec)
        if name in ("BASS_DRUM", "SNARE_DRUM", "CLICK"):
            dur = min(0.09, gap * 0.9)
        elif name in ("HARP", "PIANO", "GUITAR", "XYLOPHONE", "IRON_XYLOPHONE",
                      "BELL", "CHIME", "PLING", "BANJO", "COW_BELL", "BIT"):
            # 短音色：最多占到自己到下一个音间隔的 85%，且不超过 0.18s
            dur = min(0.18, max(0.05, gap * 0.85))
        else:
            # 长音色（长笛、贝斯、弦乐）：稍微长一点，但也不能盖住后面
            dur = min(0.26, max(0.06, gap * 0.9))

        vel = max(0.05, min(1.0, float(getattr(note, "velocity", 100) or 100) / 100.0))
        wave = _synth_note(freq, dur, sr, timbre, vel * 0.5)

        end = min(n_samples, idx + len(wave))
        mix[idx:end] += wave[: end - idx]

    peak = float(np.abs(mix).max())
    if peak > 1e-9:
        mix = mix / peak * gain

    try:
        sf.write(wav_path, mix, sr, subtype="PCM_16")
    except Exception as exc:  # noqa: BLE001
        return False, f"写出 wav 失败：{exc}"

    return True, f"{len(notes)} 个音符 · {total_sec:.1f} 秒"
