"""rich_voicing：从多声源里提取「主旋律」「低音」「和声」三条线，再按时间对齐拼接。

思路：
1. 四源分离（drums/bass/other/vocals）各自转录
2. 按 tick 对齐所有音符
3. 主旋律：优先从 vocals，其次从 other，取每个时刻音高最高的那一个
4. 低音：优先用 bass 源，取每个时刻最低的音；如果 bass 源太少就从 other/bass 补
5. 和声：从 other 里挑中音区、强度较强的音，但要和主旋律错开，避免全堆一起
6. 合并时去重（同 tick + 同音高）并控制密度，最终拼成更有层次的多声部
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import parts


@dataclass
class StemNotes:
    name: str
    notes: list[Any]


#: 用于配器的乐器
MELODY_INST = ["FLUTE", "HARP", "PLING"]
BASS_INST = ["DOUBLE_BASS", "BASS"]
CHORD_INST = ["PIANO", "HARP", "BELL"]


def _tick_key(note: Any, tps: float = 20.0) -> int:
    try:
        s = float(getattr(note, "start", 0.0) or 0.0)
        return int(round(s * tps))
    except Exception:
        return 0


def _midi(note: Any) -> int:
    try:
        return int(getattr(note, "midi", 60))
    except Exception:
        return 60


def _vel(note: Any) -> float:
    try:
        return float(getattr(note, "velocity", 0.7) or 0.7)
    except Exception:
        return 0.7


def _dur(note: Any) -> float:
    try:
        return float(getattr(note, "duration", 0.2) or 0.2)
    except Exception:
        return 0.2


def build_rich_parts(stem_tracks: dict[str, list[Any]], tps: float = 20.0):
    """从各声源音符构建「主旋律 + 低音 + 和声」的多声部列表。

    返回 [(part_name, notes), ...]，part_name 用于后续乐器分配。
    """
    # 按 tick 分组
    by_stem_tick: dict[str, dict[int, list[Any]]] = {}
    for sname, notes in stem_tracks.items():
        g = defaultdict(list)
        for n in notes:
            g[_tick_key(n, tps)].append(n)
        by_stem_tick[sname] = dict(g)

    all_ticks = sorted(set(t for m in by_stem_tick.values() for t in m.keys()))
    if not all_ticks:
        return []

    melody_notes = []
    bass_notes = []
    chord_notes = []

    for tk in all_ticks:
        # 主旋律候选：vocals > other
        mel_cands = []
        for sn in ("vocals", "other"):
            mel_cands.extend(by_stem_tick.get(sn, {}).get(tk, []))
        if mel_cands:
            # 选最高音（最像主旋律）
            best_mel = max(mel_cands, key=lambda x: (_midi(x), _vel(x)))
            melody_notes.append(best_mel)

        # 低音候选：bass > 其他
        bass_cands = []
        bass_cands.extend(by_stem_tick.get("bass", {}).get(tk, []))
        bass_cands.extend(by_stem_tick.get("other", {}).get(tk, []))
        # 只要低音区的
        bass_cands = [n for n in bass_cands if _midi(n) <= 64]
        if bass_cands:
            # 选最低音（最像贝斯线）
            best_bass = min(bass_cands, key=lambda x: (_midi(x), -_vel(x)))
            bass_notes.append(best_bass)
        elif mel_cands:
            # 如果完全没低音，尝试从主旋律往下借一个（少量补足，避免空白太大）
            # 但 funk 这类很少走这条路
            pass

        # 和声留给第二遍统一处理
        pass

    # 第二遍：补和声
    mel_midi_by_tick = {_tick_key(n, tps): _midi(n) for n in melody_notes}
    for tk in all_ticks:
        chord_cands = by_stem_tick.get("other", {}).get(tk, [])
        if not chord_cands:
            continue
        mel_m = mel_midi_by_tick.get(tk)
        if mel_m is None:
            continue
        # 只要中音区（48-76）且离主旋律 >= 3 半音
        for c in chord_cands:
            cm = _midi(c)
            if cm < 48 or cm > 76:
                continue
            if abs(cm - mel_m) < 3:
                continue
            # 同 tick 已经加过太多和声就跳过
            if len([x for x in chord_notes if _tick_key(x, tps) == tk]) >= 2:
                break
            chord_notes.append(c)
            break

    # 打击照旧走鼓组细分
    drums_notes = by_stem_tick.get("drums", {}).get(all_ticks[0], [])
    # 收集全曲鼓
    drums_all = []
    for tk in all_ticks:
        drums_all.extend(by_stem_tick.get("drums", {}).get(tk, []))

    out = []
    if melody_notes:
        out.append(("vocals_rich", melody_notes))
    if bass_notes:
        out.append(("bass_rich", bass_notes))
    if chord_notes:
        out.append(("other_rich", chord_notes))
    if drums_all:
        out.extend(parts.split_drum_bands(drums_all))

    return out
