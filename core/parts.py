"""多声部转录：把音频拆成 4 个源，各自转录并配不同乐器。

官方红乐作品动辄 8~10 种乐器、每 tick 十来个音，本质是「完整编曲」。
单条混音转录只能给出一条旋律线，这里用四源分离补上音色分工：

    drums  → BASS_DRUM / SNARE_DRUM（打击）
    bass   → DOUBLE_BASS / BASS（低音）
    other  → PIANO / GUITAR / HARP（和声与伴奏）
    vocals → FLUTE / CHIME（主旋律）

每个源单独转录后合并，乐器种类和每 tick 音符数都会明显上升。
"""

from __future__ import annotations

import numpy as np

#: 每个源对应的候选乐器（按优先级，第一个是默认）
PART_INSTRUMENTS = {
    "drums": ["BASS_DRUM", "SNARE_DRUM", "CLICK"],
    "bass": ["DOUBLE_BASS", "BASS", "DIDGERIDOO"],
    "other": ["PIANO", "GUITAR", "HARP"],
    "vocals": ["FLUTE", "CHIME", "PLING"],
}

#: 打击乐细分：按音高把 drums 轨拆成低/中/高三层，各配一种打击乐器。
#: 官方红乐里打击类能占到 20% 以上（CLICK 18% + BASS_DRUM 5.6%），
#: 只给一种鼓会让乐器种类和密度都上不去。
DRUM_BANDS = [
    # (最低 MIDI, 最高 MIDI, 乐器, 中文名)
    (0, 47, "BASS_DRUM", "底鼓"),
    (48, 59, "SNARE_DRUM", "军鼓"),
    (60, 71, "CLICK", "踩镲"),
    (72, 127, "COW_BELL", "牛铃"),
]

#: 声部中文名
PART_LABELS = {
    "drums": "打击",
    "bass": "低音",
    "other": "和声",
    "vocals": "主旋律",
}


def separate_stems(samples: np.ndarray, sr: int, chunk: int = 343980) -> dict | None:
    """用四源模型把混音拆成 drums/bass/other/vocals。

    返回 {源名: (channels, n)}，模型不可用时返回 None。
    """
    from . import vocal_separate as vs

    try:
        sess = vs._load_session("htdemucs")
    except Exception:  # noqa: BLE001
        return None

    in_name = sess.get_inputs()[0].name
    ch, total = samples.shape
    if ch == 1:
        audio = np.repeat(samples, 2, axis=0)
    else:
        audio = samples[:2].astype(np.float32)

    out = {k: np.zeros((2, total), dtype=np.float32) for k in PART_INSTRUMENTS}
    names = ["drums", "bass", "other", "vocals"]

    n_chunks = int(np.ceil(total / chunk))
    for c in range(n_chunks):
        s0 = c * chunk
        s1 = min(s0 + chunk, total)
        seg = audio[:, s0:s1]
        seg_len = seg.shape[1]
        if seg_len < chunk:
            seg = np.pad(seg, ((0, 0), (0, chunk - seg_len)), mode="constant")
        try:
            res = sess.run(None, {in_name: seg[None].astype(np.float32)})[0][0]
        except Exception:  # noqa: BLE001
            return None
        for k, nm in enumerate(names):
            out[nm][:, s0:s1] = res[k][:, :seg_len]

    return out


def transcribe_parts(
    samples: np.ndarray,
    sr: int,
    onset_threshold: float,
    frame_threshold: float,
    min_note_ms: float,
) -> list[tuple[str, list]] | None:
    """分轨转录，返回 [(声部名, 音符列表), ...]。分离失败返回 None。"""
    from .transcribe import transcribe

    stems = separate_stems(samples, sr)
    if stems is None:
        return None

    parts: list[tuple[str, list]] = []
    for name in PART_INSTRUMENTS:
        mono = stems[name].mean(axis=0)
        peak = float(np.abs(mono).max()) if mono.size else 0.0
        if peak < 1e-3:
            continue  # 这一源基本没声音，跳过
        tr = transcribe(
            mono, sr, prefer_model=True,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length_ms=min_note_ms,
        )
        if not tr.notes:
            continue

        if name == "drums":
            # 打击轨按强度+拍位拆成多个鼓组声部
            parts.extend(split_drum_bands(tr.notes))
        else:
            parts.append((name, tr.notes))
    return parts or None


def split_drum_bands(notes: list) -> list[tuple[str, list]]:
    """把打击轨拆成多个鼓组声部。

    basic-pitch 在 drums 轨上识别出的音高基本都挤在低音区，
    按音高分带会把九成音符都塞给底鼓（实测踩镲只剩 1 个）。
    改成按「强度 + 拍位」区分，更接近真实鼓组：

      强拍（tick % 8 == 0）低音量  → 底鼓 BASS_DRUM
      弱拍 + 中等强度              → 军鼓 SNARE_DRUM
      其余（高频碎音）             → 踩镲 CLICK
      特别强的重音                 → 牛铃 COW_BELL
    """
    if not notes:
        return []

    vels = sorted(float(n.velocity or 0.5) for n in notes)
    n = len(vels)
    v_low = vels[int(n * 0.35)]   # 35% 分位
    v_high = vels[int(n * 0.85)]  # 85% 分位

    buckets: dict[str, list] = {
        "BASS_DRUM": [],
        "SNARE_DRUM": [],
        "CLICK": [],
        "COW_BELL": [],
    }
    for note in notes:
        v = float(note.velocity or 0.5)
        tick = int(round(float(note.start) * 10))
        on_beat = (tick % 8) == 0        # 大约每 0.8 秒一个强拍
        half_beat = (tick % 4) == 0

        if v >= v_high and on_beat:
            buckets["COW_BELL"].append(note)
        elif on_beat and v <= v_low:
            buckets["BASS_DRUM"].append(note)
        elif half_beat:
            buckets["SNARE_DRUM"].append(note)
        else:
            buckets["CLICK"].append(note)

    out = [(f"drums_{k}", v) for k, v in buckets.items() if v]
    total = sum(len(v) for _, v in out)
    if total < len(notes):
        # 有音符没分到（理论上不会），全塞进第一种
        assigned = {id(x) for _, vs in out for x in vs}
        rest = [x for x in notes if id(x) not in assigned]
        if rest:
            if out:
                out[0] = (out[0][0], out[0][1] + rest)
            else:
                out = [("drums_BASS_DRUM", rest)]
    return out


def band_instrument(part_name: str) -> str | None:
    """drums_XXX 形式直接返回 XXX。"""
    if part_name.startswith("drums_"):
        return part_name.split("_", 1)[1]
    return None


def part_label(part_name: str) -> str:
    """声部显示名（兼容 drums_XXX 这种细分名）。"""
    if part_name in PART_LABELS:
        return PART_LABELS[part_name]
    if part_name.startswith("drums_"):
        inst = part_name.split("_", 1)[1]
        for _, _, i, label in DRUM_BANDS:
            if i == inst:
                return label
        return "打击"
    return part_name


#: 和声增强用的音程（相对根音的半音数）与乐器
HARMONY_STACK = [
    (7, "BELL"),        # 纯五度
    (12, "PLING"),      # 八度
]


def enrich_harmony(notes: list, max_extra_per_tick: int = 3) -> list[tuple[str, list]]:
    """给音符叠和声，提高每 tick 的音符密度。

    ⚠️ 这是「算式补音」，不是从音频里听出来的真实声部：
    给每个音叠加五度和八度，听感会更厚，但不保证符合原曲和声。
    只保留在音域范围内的结果，避免炸音。

    返回 [(声部名, 音符列表)]，可直接并进多声部列表。
    """
    if not notes:
        return []

    # 按 tick 分组，控制每 tick 最多补几个，防止糊成一片
    by_tick: dict[int, list] = {}
    for n in notes:
        by_tick.setdefault(int(round(float(n.start) * 10)), []).append(n)

    out: dict[str, list] = {}
    for tick, group in by_tick.items():
        # 每 tick 只给最强的那个音补和声
        strongest = max(group, key=lambda n: float(n.velocity or 0.5))
        added = 0
        for interval, inst in HARMONY_STACK:
            if added >= max_extra_per_tick:
                break
            new_midi = int(strongest.midi) + interval
            if new_midi > 96:      # 超出音符盒音域就不加
                continue
            note = type("H", (), {})()
            note.start = strongest.start
            note.duration = strongest.duration
            note.midi = new_midi
            note.velocity = float(strongest.velocity or 0.5) * 0.65   # 和声弱一些
            out.setdefault(f"harmony_{inst}", []).append(note)
            added += 1

    return [(k, v) for k, v in out.items() if v]


def harmony_instrument(part_name: str) -> str | None:
    """harmony_XXX 形式返回 XXX。"""
    if part_name.startswith("harmony_"):
        return part_name.split("_", 1)[1]
    return None


def pick_part_instrument(name: str, notes: list) -> str:
    """给某个声部选乐器：按音域在中位音高附近挑候选里最合适的。"""
    from . import nbs_writer

    cands = PART_INSTRUMENTS.get(name, ["HARP"])
    if not notes:
        return cands[0]

    midis = sorted(int(n.midi) for n in notes)
    median = midis[len(midis) // 2]

    if name == "drums":
        return cands[0]
    if name == "bass":
        return "DOUBLE_BASS" if median < 48 else "BASS"
    if name == "vocals":
        return "FLUTE" if median > 72 else "CHIME"
    # other
    if median < 60:
        return "GUITAR"
    if median < 72:
        return "PIANO"
    return nbs_writer.normalize_instrument("HARP")
