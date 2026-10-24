"""Minecraft 音符盒紧凑布局 → .litematic 结构文件。

移植自 nbs-fractal (crates.io 0.1.0) 的 `schematic/compact.rs` 与
`schematic/blocks.rs`，并按 rustmatica 的 NBT schema 打包成 .litematic。

坐标约定与 Rust 版一致：x=东(east)、y=高(elevation)、z=南(south)。
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from .nbt import IntArray, ListTag, Long, LongArray, TAG_COMPOUND, TAG_END, gzip_bytes, write_nbt

# ---------------------------------------------------------------------------
# 乐器表（对应 Rust 的 Instrument::note_property / block_resource）
# ---------------------------------------------------------------------------

INSTRUMENTS: dict[str, tuple[str, str]] = {
    # 键: (note_property, 乐器方块 Name)
    "HARP": ("harp", "minecraft:dirt"),
    "DOUBLE_BASS": ("bass", "minecraft:oak_planks"),
    "BASS_DRUM": ("basedrum", "minecraft:stone"),
    "SNARE_DRUM": ("snare", "minecraft:sand"),
    "CLICK": ("hat", "minecraft:glass"),
    "GUITAR": ("guitar", "minecraft:white_wool"),
    "FLUTE": ("flute", "minecraft:clay"),
    "BELL": ("bell", "minecraft:gold_block"),
    "CHIME": ("chime", "minecraft:packed_ice"),
    "XYLOPHONE": ("xylophone", "minecraft:bone_block"),
    "IRON_XYLOPHONE": ("iron_xylophone", "minecraft:iron_block"),
    "COW_BELL": ("cow_bell", "minecraft:soul_sand"),
    "DIDGERIDOO": ("didgeridoo", "minecraft:pumpkin"),
    "BIT": ("bit", "minecraft:emerald_block"),
    "BANJO": ("banjo", "minecraft:hay_block"),
    "PLING": ("pling", "minecraft:glowstone"),
    "TRUMPET": ("trumpet", "minecraft:waxed_copper_block"),
    "TRUMPET_EXPOSED": ("trumpet_exposed", "minecraft:waxed_exposed_copper"),
    "TRUMPET_WEATHERED": ("trumpet_weathered", "minecraft:waxed_weathered_copper"),
    "TRUMPET_OXIDIZED": ("trumpet_oxidized", "minecraft:waxed_oxidized_copper"),
}

# NBS 音高 → Minecraft 音符
NBS_KEY_MIN = 33  # F#3
NBS_KEY_MAX = 57  # F#5
MIDI_OFFSET_NBS = 21  # nbs_key = midi - 21

# Rust 的 u64 MAX
U64_MAX = 0xFFFFFFFFFFFFFFFF


def nbs_key_to_midi(nbs_key: int) -> int:
    """NBS 音高 → 标准 MIDI 音高。"""
    return nbs_key + MIDI_OFFSET_NBS


def midi_to_nbs_key(midi: int) -> int:
    """标准 MIDI 音高 → NBS 音高。"""
    return midi - MIDI_OFFSET_NBS


def key_to_mc_note(nbs_key: int) -> int | None:
    """NBS 音高(33..57) → Minecraft 音符(0..24)。超出范围返回 None。"""
    note = nbs_key - NBS_KEY_MIN
    return note if 0 <= note <= 24 else None


def fold_to_nbs_range(midi: int) -> int:
    """把任意 MIDI 音高折叠进 Minecraft 音符盒可演奏的范围(F#3..F#5)。

    做法是按八度平移（保持音级），再夹到边界。
    """
    nbs = midi_to_nbs_key(midi)
    if nbs < NBS_KEY_MIN:
        nbs += 12 * ((NBS_KEY_MIN - nbs + 11) // 12)
    if nbs > NBS_KEY_MAX:
        nbs -= 12 * ((nbs - NBS_KEY_MAX + 11) // 12)
    return max(NBS_KEY_MIN, min(NBS_KEY_MAX, nbs))


# ---------------------------------------------------------------------------
# 方块状态
# ---------------------------------------------------------------------------

FACING_EAST = "east"
FACING_WEST = "west"
FACING_NORTH = "north"
FACING_SOUTH = "south"

_INVERT = {
    FACING_EAST: FACING_WEST,
    FACING_WEST: FACING_EAST,
    FACING_NORTH: FACING_SOUTH,
    FACING_SOUTH: FACING_NORTH,
}


def _state(name: str, properties: dict[str, str] | None = None) -> dict[str, Any]:
    return {"Name": name, "Properties": properties or {}}


def air() -> dict[str, Any]:
    return _state("minecraft:air")


def chain_block() -> dict[str, Any]:
    """链条方块基座（smooth_stone）。"""
    return _state("minecraft:smooth_stone")


def redstone_wire() -> dict[str, Any]:
    return _state(
        "minecraft:redstone_wire",
        {"power": "0", "north": "side", "south": "side", "east": "side", "west": "side"},
    )


def repeater(delay: int, facing: str, powered: bool = False, locked: bool = False) -> dict[str, Any]:
    """中继器。注意 Rust 版里 facing 是反转存储的。"""
    return _state(
        "minecraft:repeater",
        {
            "delay": str(delay),
            "facing": _INVERT[facing],
            "locked": "true" if locked else "false",
            "powered": "true" if powered else "false",
        },
    )


def note_block_state(tone: tuple[str, int] | None) -> dict[str, Any] | None:
    """音符盒方块状态。tone=None 返回 None。"""
    if tone is None:
        return None
    instrument, nbs_key = tone
    note = key_to_mc_note(nbs_key)
    if note is None:
        return None
    prop, _ = INSTRUMENTS.get(instrument, INSTRUMENTS["HARP"])
    return _state(
        "minecraft:note_block",
        {"note": str(note), "powered": "false", "instrument": prop},
    )


def instrument_block_state(tone: tuple[str, int] | None) -> dict[str, Any] | None:
    """音符下方的乐器方块。tone=None 返回 None。"""
    if tone is None:
        return None
    instrument, nbs_key = tone
    if key_to_mc_note(nbs_key) is None:
        return None
    entry = INSTRUMENTS.get(instrument)
    if entry is None:
        return None
    _prop, block = entry
    if instrument == "BANJO":
        return _state(block, {"axis": "y"})
    return _state(block)


def note_block(tone: tuple[str, int] | None, fallback) -> dict[str, Any]:
    st = note_block_state(tone)
    return st if st is not None else fallback()


def inst_block(tone: tuple[str, int] | None, fallback) -> dict[str, Any]:
    st = instrument_block_state(tone)
    return st if st is not None else fallback()


# ---------------------------------------------------------------------------
# Tile：stem/canopy 一对方块
# ---------------------------------------------------------------------------

# Tile 种类
T_DELAY = "delay"
T_LINK = "link"
T_TERMINAL = "terminal"
T_NODE = "node"
T_TURNING_DELAY = "turning_delay"
T_TURNING_LINK = "turning_link"
T_TURNING_TERMINAL = "turning_terminal"
T_TURNING_NODE = "turning_node"


class Tile:
    """一个 stem-canopy 瓦片对。"""

    __slots__ = ("kind", "delay", "tones")

    def __init__(self, kind: str, delay: int = 0, tones: tuple = ()):
        self.kind = kind
        self.delay = delay
        self.tones = tones

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Tile({self.kind}, delay={self.delay}, tones={self.tones})"

    @staticmethod
    def stem(delay: int, is_turning: bool) -> "Tile":
        if delay == 0:
            return Tile(T_TURNING_LINK if is_turning else T_LINK)
        return Tile(T_TURNING_DELAY if is_turning else T_DELAY, delay=delay)

    @staticmethod
    def canopy(notes: list, is_turning: bool, is_terminal: bool) -> "Tile":
        """按 Rust 的 `iter::from_fn(|| notes.pop())` 语义从**列表尾部**取用音符。

        注意：notes 会被原地修改（弹出已消费的音符），调用方需继续使用同一列表。
        """
        def take() -> tuple[str, int] | None:
            return notes.pop() if notes else None

        if is_turning and is_terminal:
            return Tile(T_TURNING_TERMINAL, tones=(take(), take()))
        if is_turning:
            return Tile(T_TURNING_NODE, tones=(take(),))
        if is_terminal:
            return Tile(T_TERMINAL, tones=(take(), take(), take()))
        return Tile(T_NODE, tones=(take(), take()))

    def get_block(self, layout_index: int, repeater_facing: str) -> dict[str, Any] | None:
        """layout_index 0..7，对应 Rust 的 Tile::get_block。"""
        k = self.kind
        if k == T_DELAY:
            if layout_index == 0:
                return chain_block()
            if layout_index == 1:
                return repeater(self.delay, repeater_facing)
            return None
        if k == T_LINK:
            if layout_index == 0:
                return chain_block()
            if layout_index == 1:
                return redstone_wire()
            return None
        if k == T_TERMINAL:
            center, left, right = self.tones
            if layout_index == 0:
                return inst_block(center, chain_block)
            if layout_index == 1:
                return note_block(center, chain_block)
            if layout_index == 3:
                return inst_block(left, air)
            if layout_index == 4:
                return note_block(left, air)
            if layout_index == 6:
                return inst_block(right, air)
            if layout_index == 7:
                return note_block(right, air)
            return None
        if k == T_NODE:
            left, right = self.tones
            if layout_index in (0, 1):
                return chain_block()
            if layout_index == 2:
                return redstone_wire()
            if layout_index == 3:
                return inst_block(left, air)
            if layout_index == 4:
                return note_block(left, air)
            if layout_index == 6:
                return inst_block(right, air)
            if layout_index == 7:
                return note_block(right, air)
            return None
        if k == T_TURNING_DELAY:
            if layout_index in (0, 3):
                return chain_block()
            if layout_index == 1:
                return redstone_wire()
            if layout_index == 4:
                return repeater(self.delay, repeater_facing)
            return None
        if k == T_TURNING_LINK:
            if layout_index in (0, 3):
                return chain_block()
            if layout_index in (1, 4):
                return redstone_wire()
            return None
        if k == T_TURNING_TERMINAL:
            center, side = self.tones
            if layout_index == 0:
                return inst_block(center, chain_block)
            if layout_index == 1:
                return note_block(center, chain_block)
            if layout_index == 3:
                return inst_block(side, air)
            if layout_index == 4:
                return note_block(side, air)
            return None
        if k == T_TURNING_NODE:
            (side,) = self.tones
            if layout_index in (0, 1):
                return chain_block()
            if layout_index == 2:
                return redstone_wire()
            if layout_index == 3:
                return inst_block(side, air)
            if layout_index == 4:
                return note_block(side, air)
            return None
        return None


# ---------------------------------------------------------------------------
# Track：一轨的瓦片序列
# ---------------------------------------------------------------------------


class Track:
    """一轨的 tiles 与行列元数据。"""

    def __init__(self, tiles: list[Tile], cols: int | None):
        self.tiles = tiles
        self.cols = cols  # None 表示不换行

    def __len__(self) -> int:
        return len(self.tiles)

    def rows(self) -> int:
        if self.cols is None:
            return 1
        return -(-len(self) // self.cols)

    def cols_or_len(self) -> int:
        if self.cols is None:
            return len(self)
        return self.cols

    def get_tile(self, row: int, offset: int) -> Tile | None:
        idx = row * self.cols_or_len() + offset
        if 0 <= idx < len(self.tiles):
            return self.tiles[idx]
        return None

    def at_row_start(self) -> bool:
        if self.cols is None:
            return len(self) < 2
        return len(self) % self.cols == 0

    def at_row_end(self) -> bool:
        if self.cols is None:
            return False
        return (len(self) + 2) % self.cols == 0

    def tile_block(self, row: int, col: int, layout_idx: int) -> dict[str, Any] | None:
        """取某行某列的方块，并按行列选择中继器朝向。"""
        if (row & 1) == 0:
            facing = FACING_SOUTH
        else:
            facing = FACING_NORTH
        if col < 2:
            facing = FACING_EAST
        tile = self.get_tile(row, col)
        if tile is None:
            return None
        return tile.get_block(layout_idx, facing)


def _place_delay(
    stem_delay: int, canopy_delay: int, at_start: bool, at_end: bool
) -> tuple[Tile, Tile, int]:
    stem = Tile.stem(stem_delay, at_start)
    if canopy_delay == 0:
        canopy = Tile.canopy([], at_start, not at_end)
    else:
        canopy = Tile.stem(canopy_delay, False)
    return stem, canopy, stem_delay + canopy_delay


def _pop_delay(delay: int, coarse: int, track: Track):
    """从 Rust 移植的延迟填充逻辑。返回 (stem, canopy, consume) 或 None。"""
    last = track.tiles[-1] if track.tiles else None
    chain = last is not None and last.kind == T_DELAY and last.delay == coarse
    at_start = track.at_row_start()
    at_end = track.at_row_end()

    def pair(stem: int, canopy: int):
        return _place_delay(stem, canopy, at_start, at_end)

    def turn(stem: int):
        return _place_delay(stem, 0, at_start, at_end)

    # 严格按 Rust match 的顺序：首个命中的分支生效
    if 2 <= coarse <= 4:
        if (not chain) and (not at_start) and delay == coarse * 3:
            return pair(coarse, 0)
        if (not at_start) and (not at_end) and delay > coarse * 2:
            return pair(coarse, coarse)
        if chain and (not at_start) and delay == coarse * 2:
            return pair(coarse, 1)
        if chain and (not at_start) and delay >= coarse:
            return pair(coarse - 1, 0)
        if (not chain) and (not at_start) and delay > coarse:
            return pair(coarse, 0)
        if (not chain) and at_start and delay > coarse:
            return turn(coarse)
    elif coarse == 1:
        if (not chain) and at_start and delay > 1:
            return turn(coarse)
        if (not chain) and (not at_start) and delay > 1:
            return pair(coarse, 0)

    if at_start and delay > 4:
        return turn(4)
    if (not at_start) and delay > 8:
        return pair(4, 4)
    if (not at_start) and delay > 4:
        return pair(4, 0)
    return None


def build_track(
    timed_notes: Iterable[tuple[int, list[tuple[str, int]]]],
    repeater_coarse: int | None,
    columns: int | None,
) -> Track:
    """移植 Track::new。

    timed_notes: (红石刻, 音符列表)，按时间升序。
    repeater_coarse: None 表示不限（对应 Rust 的 Option<NonZero> = None）。
    columns: 换行长度（None = 不换行）。
    """
    coarse = U64_MAX if repeater_coarse is None else repeater_coarse
    cols = None if columns is None else columns * 2
    tiles: list[Tile] = []
    track = Track(tiles, cols)
    current_tick = U64_MAX

    for redstone_tick, notes in timed_notes:
        pool = list(notes)
        delay = (redstone_tick - current_tick) & U64_MAX
        current_tick = redstone_tick

        while True:
            res = _pop_delay(delay, coarse, track)
            if res is None:
                break
            stem, canopy, consume = res
            track.tiles.append(stem)
            track.tiles.append(canopy)
            delay -= consume

        at_start = track.at_row_start()
        at_end = track.at_row_end()
        stem = Tile.stem(delay, at_start)
        canopy = Tile.canopy(pool, at_start, not at_end)
        track.tiles.append(stem)
        track.tiles.append(canopy)

        if pool:
            at_start = track.at_row_start()
            at_end = track.at_row_end()
            is_terminal = (not at_end) and at_start and len(pool) <= 2
            stem = Tile.stem(0, at_start)
            canopy = Tile.canopy(pool, at_start, is_terminal)
            track.tiles.append(stem)
            track.tiles.append(canopy)

        while pool:
            at_start = track.at_row_start()
            at_end = track.at_row_end()
            limit = 2 if at_start else 3
            is_terminal = (not at_end) and len(pool) <= limit
            stem = Tile.stem(0, at_start)
            canopy = Tile.canopy(pool, at_start, is_terminal)
            track.tiles.append(stem)
            track.tiles.append(canopy)

    return track


# ---------------------------------------------------------------------------
# CompactLayout：单条紧凑线
# ---------------------------------------------------------------------------

ELEVATION = 3


class CompactLayout:
    """一条紧凑轨道（红石刻粒度）。"""

    def __init__(self, track: Track):
        self.track = track
        self.easting = track.rows() * 2 + 1
        self.southing = track.cols_or_len()

    def size(self) -> tuple[int, int, int]:
        return (self.easting, ELEVATION, self.southing)

    def block_at(self, x: int, y: int, z: int) -> dict[str, Any] | None:
        """x=东(east), y=高(elevation), z=南(southing)。"""
        southing = z
        easting = x
        elevation = y

        def tile_col(s: int, row: int) -> int:
            return s + 1 if (row & 1) == 0 else self.southing - s

        if southing == 0:
            # 北边缘转弯
            e = easting + 1
            group = e & 3
            row = (e // 4) * 2
            col = group // 2
            layout_idx = elevation + (group & 1) * 3
            return self.track.tile_block(row, col, layout_idx)
        if southing + 1 == self.southing:
            # 南边缘转弯
            e = easting + 3
            group = e & 3
            row = (e // 4) * 2 - 1
            col = group // 2
            layout_idx = elevation + (group & 1) * 3
            return self.track.tile_block(row, col, layout_idx)
        if easting & 1 == 1:
            # 主干行
            row = easting // 2
            col = tile_col(southing, row)
            return self.track.tile_block(row, col, elevation)
        # 齿行
        cell = easting // 2
        zig = (cell + southing) & 1
        row = cell - zig
        col = tile_col(southing, row)
        layout_idx = elevation + 3 + zig * 3
        return self.track.tile_block(row, col, layout_idx)

    def blocks(self) -> dict[tuple[int, int,int], dict[str, Any]]:
        sx, sy, sz = self.size()
        out: dict[tuple[int, int, int], dict[str, Any]] = {}
        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    st = self.block_at(x, y, z)
                    if st is not None:
                        out[(x, y, z)] = st
        return out


# ---------------------------------------------------------------------------
# MultiCompactLayout：多轨排布
# ---------------------------------------------------------------------------


def split_even_odd(
    notes: dict[int, list[tuple[str, int]]],
    coarse: int | None,
) -> list[tuple[dict[int, list[tuple[str, int]]], int | None]]:
    """按游戏刻奇偶拆成两条红石刻子轨。

    对应 Rust 的 MultiCompactLayout::split_even_odd。
    """
    buckets: list[dict[int, list[tuple[str, int]]]] = [{}, {}]
    for game_tick in sorted(notes):
        tones = notes[game_tick]
        if not tones:
            continue
        idx = game_tick % 2
        key = game_tick // 2
        buckets[idx].setdefault(key, []).extend(tones)
    return [(b, coarse) for b in buckets if b]


def build_compact_blocks(
    notes: dict[int, list[tuple[str, int]]],
    coarse: int = 0,
    wrap: int = 0,
    gap: int = 0,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """生成紧凑布局的全部方块。

    notes: {游戏刻(20tps): [(乐器名, NBS音高), ...]}
    coarse: 0 表示不限（1..=4 表示中继器延迟粒度）
    wrap: 每行最多多少列（0 = 不换行）
    gap: 相邻轨道间距
    """
    coarse_opt = coarse if coarse and coarse > 0 else None
    wrap_opt = wrap if wrap and wrap > 0 else None

    subtracks = split_even_odd(notes, coarse_opt)

    layouts: list[CompactLayout] = []
    for sub, c in subtracks:
        timed = [(tick, tones) for tick, tones in sorted(sub.items())]
        track = build_track(timed, c, wrap_opt)
        layouts.append(CompactLayout(track))

    # Arranged::new(layouts, Axis::Easting, gap)
    out: dict[tuple[int, int, int], dict[str, Any]] = {}
    cursor_x = -gap
    extent = [0, 0, 0]
    for layout in layouts:
        sx, sy, sz = layout.size()
        anchor_x = cursor_x + gap
        cursor_x = anchor_x + sx
        extent[0] = max(extent[0], sx)
        extent[1] = max(extent[1], sy)
        extent[2] = max(extent[2], sz)
        for (lx, ly, lz), st in layout.blocks().items():
            out[(lx + anchor_x, ly, lz)] = st

    size = (
        max(max(cursor_x, 0), extent[0]),
        max(max(0, 0), extent[1]),
        max(max(0, 0), extent[2]),
    )
    return out


# ---------------------------------------------------------------------------
# .litematic 打包
# ---------------------------------------------------------------------------

AIR = {"Name": "minecraft:air", "Properties": {}}


def _num_bits(palette_len: int) -> int:
    bits = 2
    while (1 << bits) < palette_len:
        bits += 1
    return bits


def _pack_blocks(indices: list[int], bits: int) -> LongArray:
    """把 palette 索引按 rustmatica 的规则打包成 LongArray。"""
    flat: list[int] = []
    for idx in indices:
        for bit in range(bits):
            flat.append((idx >> bit) & 1)
    out: list[int] = []
    for start in range(0, len(flat), 64):
        chunk = flat[start : start + 64]
        acc = 0
        for b in reversed(chunk):
            acc = (acc << 1) | b
        out.append(acc)
    return LongArray(out)


def to_litematic(
    blocks: dict[tuple[int, int, int], dict[str, Any]],
    name: str = "song",
    description: str = "",
    author: str = "mp3-nbs-converter",
    minecraft_data_version: int = 2975,
    region_name: str = "Note Block Track Schematic",
) -> bytes:
    """把方块集合打包成 .litematic（gzip 压缩的 NBT）。

    返回可直接写入文件的字节。
    """
    if not blocks:
        size = (1, 1, 1)
        origin = (0, 0, 0)
    else:
        xs = [p[0] for p in blocks]
        ys = [p[1] for p in blocks]
        zs = [p[2] for p in blocks]
        origin = (min(xs), min(ys), min(zs))
        size = (max(xs) - origin[0] + 1, max(ys) - origin[1] + 1, max(zs) - origin[2] + 1)

    ox, oy, oz = origin
    sx, sy, sz = size
    volume = sx * sy * sz

    # palette：index 0 固定是空气
    palette: list[dict[str, Any]] = [AIR]
    palette_index: dict[str, int] = {}

    def key_of(st: dict[str, Any]) -> str:
        props = st.get("Properties") or {}
        return st["Name"] + "|" + ";".join(f"{k}={v}" for k, v in sorted(props.items()))

    indices = [0] * volume
    for (x, y, z), st in blocks.items():
        lx, ly, lz = x - ox, y - oy, z - oz
        if not (0 <= lx < sx and 0 <= ly < sy and 0 <= lz < sz):
            continue
        k = key_of(st)
        idx = palette_index.get(k)
        if idx is None:
            idx = len(palette)
            palette_index[k] = idx
            palette.append(st)
        # rustmatica 的 pos_to_index: x + y*size.x*size.z + z*size.x
        indices[lx + ly * sx * sz + lz * sx] = idx

    bits = _num_bits(len(palette))
    block_states = _pack_blocks(indices, bits)

    region = {
        "Position": {"x": ox, "y": oy, "z": oz},
        "Size": {"x": sx, "y": sy, "z": sz},
        "BlockStatePalette": ListTag(TAG_COMPOUND, palette),
        "BlockStates": block_states,
        "TileEntities": ListTag(TAG_END, []),
        "Entities": ListTag(TAG_END, []),
        "PendingBlockTicks": ListTag(TAG_END, []),
        "PendingFluidTicks": ListTag(TAG_END, []),
    }

    # Litematica 的 timeCreated/timeModified 是 64 位毫秒时间戳
    now_ms = Long(int(time.time() * 1000))
    metadata = {
        "Name": name,
        "Author": author,
        "Description": description,
        "RegionCount": 1,
        "TotalVolume": volume,
        "TotalBlocks": sum(1 for i in indices if i != 0),
        "TimeCreated": now_ms,
        "TimeModified": now_ms,
        "EnclosingSize": {"x": sx, "y": sy, "z": sz},
        # PreviewImageData 是可选的预览图字节数组，这里留空
        "PreviewImageData": bytearray(),
    }

    root = {
        "Regions": {region_name: region},
        "MinecraftDataVersion": minecraft_data_version,
        "Version": 6,
        "SubVersion": 1,
        "Metadata": metadata,
    }
    return gzip_bytes(write_nbt(root))


def build_litematic_from_notes(
    notes: dict[int, list[tuple[str, int]]],
    coarse: int = 0,
    wrap: int = 0,
    gap: int = 0,
    name: str = "song",
    description: str = "",
    author: str = "mp3-nbs-converter",
) -> bytes:
    """build_compact_blocks + to_litematic 的便捷封装。"""
    blocks = build_compact_blocks(notes, coarse=coarse, wrap=wrap, gap=gap)
    return to_litematic(blocks, name=name, description=description, author=author)
