"""从音符列表生成标准 MIDI 文件（SMF Type 1）。

使用 mido 写出，多轨道结构：
  轨 0：conductor（tempo / time signature / track name）
  轨 1..n：各条音轨
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mido


@dataclass
class MidiNote:
    """一条 MIDI 音符。"""

    start: float        # 秒
    duration: float     # 秒
    midi: int           # 0..127
    velocity: int = 100
    channel: int = 0


@dataclass
class MidiTrack:
    """一条 MIDI 轨。"""

    name: str
    notes: list[MidiNote] = field(default_factory=list)
    program: int | None = None   # GM 音色编号
    channel: int = 0


# 乐器 id → (GM program 编号, 轨名)
MIDI_PROGRAMS: dict[str, tuple[int, str]] = {
    "HARP": (0, "Harp"),
    "PIANO": (0, "Piano"),
    "BELL": (14, "Tubular Bells"),
    "FLUTE": (73, "Flute"),
    "GUITAR": (24, "Acoustic Guitar"),
    "CHIME": (14, "Chimes"),
    "XYLOPHONE": (13, "Marimba"),
    "IRON_XYLOPHONE": (13, "Marimba"),
    "COW_BELL": (113, "Cowbell"),
    "DIDGERIDOO": (122, "Didgeridoo"),
    "BIT": (117, "Toy"),
    "BANJO": (105, "Banjo"),
    "PLING": (117, "Toy"),
    "TRUMPET": (56, "Trumpet"),
    "DOUBLE_BASS": (42, "Contrabass"),
    "BASS_DRUM": (118, "Percussion"),
    "SNARE_DRUM": (119, "Percussion"),
    "CLICK": (118, "Percussion"),
    "STRINGS": (48, "Strings"),
}


def _midi_safe(text: str, fallback: str = "") -> str:
    """MIDI 的文本元事件是 latin-1 编码，中文无法写入，这里做安全转写。"""
    if not text:
        return fallback
    out = []
    for ch in text:
        try:
            ch.encode("latin-1")
        except UnicodeEncodeError:
            out.append("?")
        else:
            out.append(ch)
    result = "".join(out).strip()
    return result or fallback


def write_midi_file(
    path: str,
    tracks: list[MidiTrack],
    song_name: str = "song",
    bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
) -> str:
    """写出 MIDI 文件，返回路径。"""
    mid = mido.MidiFile(type=1)

    # --- 轨 0：conductor ---
    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("track_name", name=_midi_safe(song_name, "song"), time=0))
    conductor.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    num, den = time_signature
    conductor.append(
        mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0)
    )
    mid.tracks.append(conductor)

    # --- 音轨 ---
    for tr in tracks:
        tk = mido.MidiTrack()
        tk.append(mido.MetaMessage("track_name", name=_midi_safe(tr.name, "Track"), time=0))
        if tr.program is not None:
            tk.append(
                mido.Message("program_change", program=tr.program, channel=tr.channel, time=0)
            )
        events: list[tuple[float, int, int, mido.Message]] = []
        for n in tr.notes:
            vel = max(1, min(127, int(n.velocity)))
            ch = n.channel if n.channel is not None else tr.channel
            pitch = max(0, min(127, int(n.midi)))
            events.append((float(n.start), 1, pitch, mido.Message("note_on", note=pitch, velocity=vel, time=0, channel=ch)))
            events.append(
                (float(n.start) + max(0.01, float(n.duration)), 0, pitch,
                 mido.Message("note_off", note=pitch, velocity=0, time=0, channel=ch))
            )
        # 同一时刻先关后开，避免同音高叠加被切断
        events.sort(key=lambda e: (e[0], e[1]))
        prev = 0.0
        for t, _, _, msg in events:
            delta = t - prev
            prev = t
            # MIDI 的 delta time 是整数毫秒
            tk.append(msg.copy(time=max(0, int(round(delta * 1000)))))
        tk.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(tk)

    mid.save(path)
    return path
