"""从音符列表生成 .nbs（Note Block Studio v5）文件。

使用 pynbs 写出，格式细节与 Note Block Studio 兼容。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pynbs

# NBS 音高范围与 Minecraft 音符盒可演奏范围
NBS_KEY_MIN = 0
NBS_KEY_MAX = 87
MC_KEY_MIN = 33   # F#3
MC_KEY_MAX = 57   # F#5

# NBS 乐器 id → (显示名, 采样文件名, pitch, press_key)
# pitch 是该乐器相对 C4(=60) 的偏移采样速率，Note Block Studio 惯例值
NBS_INSTRUMENTS: dict[str, tuple[str, str, int, bool]] = {
    "HARP": ("Harp", "harp.wav", 45, True),
    "BASS": ("Bass", "bass.wav", 45, True),
    "BELL": ("Bell", "bell.wav", 45, True),
    "FLUTE": ("Flute", "flute.wav", 45, True),
    "GUITAR": ("Guitar", "guitar.wav", 45, True),
    "CHIME": ("Chime", "chime.wav", 45, True),
    "XYLOPHONE": ("Xylophone", "xylophone.wav", 45, True),
    "IRON_XYLOPHONE": ("Iron Xylophone", "iron_xylophone.wav", 45, True),
    "COW_BELL": ("Cow Bell", "cow_bell.wav", 45, True),
    "DIDGERIDOO": ("Didgeridoo", "didgeridoo.wav", 45, True),
    "BIT": ("Bit", "bit.wav", 45, True),
    "BANJO": ("Banjo", "banjo.wav", 45, True),
    "PLING": ("Pling", "pling.wav", 45, True),
    "TRUMPET": ("Trumpet", "trumpet.wav", 45, True),
    "DOUBLE_BASS": ("Double Bass", "double_bass.wav", 45, True),
    "BASS_DRUM": ("Bass Drum", "bass_drum.wav", 45, True),
    "SNARE_DRUM": ("Snare Drum", "snare_drum.wav", 45, True),
    "CLICK": ("Click", "click.wav", 45, True),
    "PIANO": ("Piano", "piano.wav", 45, True),
    "STRINGS": ("Strings", "strings.wav", 45, True),
}

# 乐器别名 → 标准 id
ALIASES = {
    "harp": "HARP",
    "piano": "PIANO",
    "flute": "FLUTE",
    "bell": "BELL",
    "double_bass": "DOUBLE_BASS",
    "guitar": "GUITAR",
    "bass_drum": "BASS_DRUM",
    "snare_drum": "SNARE_DRUM",
    "click": "CLICK",
    "banjo": "BANJO",
    "trumpet": "TRUMPET",
    "cow_bell": "COW_BELL",
    "xylophone": "XYLOPHONE",
    "iron_xylophone": "IRON_XYLOPHONE",
    "bit": "BIT",
    "pling": "PLING",
    "didgeridoo": "DIDGERIDOO",
    "chime": "CHIME",
    "bass": "BASS",
    "strings": "STRINGS",
}


def normalize_instrument(name: str) -> str:
    """把任意写法归一成标准乐器 id。"""
    if not name:
        return "HARP"
    key = name.strip().replace(" ", "_").replace("-", "_").upper()
    if key in NBS_INSTRUMENTS:
        return key
    return ALIASES.get(key.lower(), "HARP")


def _nbs_safe(text: str, fallback: str = "") -> str:
    """NBS 字符串字段按 cp1252 存储（Note Block Studio 的 legacy 编码）。

    非 cp1252 字符（中文等）无法写入，硬写会抛 UnicodeEncodeError；
    这里统一丢弃并用 fallback 兜底，保证文件可被所有工具读回。
    """
    if not text:
        return fallback
    out = []
    for ch in text:
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            continue  # 丢弃非 cp1252 字符
        out.append(ch)
    result = "".join(out).strip()
    return result or fallback


@dataclass
class Note:
    """一条音符。"""

    tick: int          # NBS tick（10 tick/秒）
    layer: int
    key: int           # NBS 音高 0..87
    velocity: int = 100
    panning: int = 0
    pitch: int = 0
    instrument: str = "HARP"

    @property
    def duration(self) -> int:
        return max(1, self._duration)

    _duration: int = 1


@dataclass
class Track:
    """一条轨道（一层）。"""

    name: str
    notes: list[Note] = field(default_factory=list)
    volume: int = 100
    panning: int = 0
    lock: bool = False


def write_nbs_file(
    path: str,
    tracks: list[Track],
    song_name: str = "song",
    author: str = "mp3-nbs-converter",
    original_author: str = "",
    description: str = "",
    tempo: float = 10.0,
    time_signature: int = 4,
) -> str:
    """写出 .nbs 文件，返回路径。"""
    # 收集所有用到的乐器
    used: list[str] = []
    for tr in tracks:
        for n in tr.notes:
            inst = normalize_instrument(n.instrument)
            if inst not in used:
                used.append(inst)
    if not used:
        used = ["HARP"]

    f = pynbs.new_file(
        song_name=_nbs_safe(song_name, "song"),
        song_author=_nbs_safe(author, ""),
        original_author=_nbs_safe(original_author, ""),
        description=_nbs_safe(description, ""),
        tempo=tempo,
        time_signature=time_signature,
    )
    # NBS 的 tempo 是"每秒 tick 数"，pynbs new_file 的 tempo 参数语义同上
    f.header.tempo = int(round(tempo))

    # new_file 自带一个占位 Layer(0)，清空后按实际轨道重建
    f.layers.clear()

    for i, inst in enumerate(used):
        disp, fname, pitch, press = NBS_INSTRUMENTS[inst]
        f.instruments.append(pynbs.Instrument(i, disp, fname, pitch, press))

    # NBS 格式规定：同一 tick 同一层只能有一个音符。
    # 和声（同 tick 多个音高）必须分配到额外的层，否则读取方会把
    # 「层跳变 = 0」误判成层列表结束，导致文件损坏。
    # 这里按需新建层来消解冲突，保证所有音符都能保留。
    layer_names: list[str] = []
    layer_owners: list[str] = []     # 每层属于哪条原始轨道
    slot_used: dict[tuple[int, int], int] = {}   # (tick, layer) -> 音符
    pending: list[tuple[int, int, int, str, int, int, int, int]] = []
    # 每条字段：(tick, layer, key, instrument, velocity, panning, pitch, orig_track)

    for tr_idx, tr in enumerate(tracks):
        base_layer = len(layer_names)
        layer_names.append(_nbs_safe(tr.name, f"Layer {base_layer}"))
        layer_owners.append(tr.name)
        for n in tr.notes:
            key = max(NBS_KEY_MIN, min(NBS_KEY_MAX, int(n.key)))
            inst_idx = used.index(normalize_instrument(n.instrument))
            tick = int(n.tick)
            velocity = max(1, min(127, int(n.velocity)))
            panning = max(-64, min(63, int(n.panning)))
            pitch = int(n.pitch)

            layer = base_layer
            while (tick, layer) in slot_used:
                # 该 tick 在这层已有音符 → 找下一个可用层
                layer += 1
                while layer >= len(layer_names):
                    layer_names.append(
                        _nbs_safe(f"{tr.name} {layer - base_layer + 1}", f"Layer {layer}")
                    )
                    layer_owners.append(tr.name)
            slot_used[(tick, layer)] = 1
            pending.append((tick, layer, key, tr.name, inst_idx, velocity, panning, pitch))

    for layer_idx, name in enumerate(layer_names):
        f.layers.append(pynbs.Layer(layer_idx, name, False, 100, 0))

    max_tick = 0
    for tick, layer, key, _tname, inst_idx, velocity, panning, pitch in pending:
        f.notes.append(
            pynbs.Note(
                tick=tick,
                layer=layer,
                instrument=inst_idx,
                key=key,
                velocity=velocity,
                panning=panning,
                pitch=pitch,
            )
        )
        max_tick = max(max_tick, tick)

    # NBS 要求音符按 tick 排序（同 tick 内按 layer 排序）
    f.notes.sort(key=lambda n: (n.tick, n.layer))

    f.header.song_length = max_tick + 20
    f.save(path)
    return path
