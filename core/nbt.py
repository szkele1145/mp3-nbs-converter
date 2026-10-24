"""极简 NBT 读写实现（big-endian），用于生成 .litematic。

只实现 .litematic 需要的标签类型，避免引入 nbtlib 等外部依赖。
"""

from __future__ import annotations

import gzip
import struct
from typing import Any

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class IntArray(list):
    """TAG_Int_Array。元素按有符号 32 位存储。"""


class LongArray(list):
    """TAG_Long_Array。元素按有符号 64 位存储。"""


class Long(int):
    """TAG_Long（64 位有符号整数）。

    用于像 Litematica 的 timeCreated 这类毫秒级时间戳——
    它们放不进 32 位 int。
    """


class ListTag(list):
    """TAG_List，额外记录元素类型。"""

    def __init__(self, item_type: int, items: Any = ()):
        super().__init__(items)
        self.item_type = item_type


def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v


def _i64(v: int) -> int:
    v &= 0xFFFFFFFFFFFFFFFF
    return v - 0x10000000000000000 if v >= 0x8000000000000000 else v


def _write_string(buf: bytearray, s: str) -> None:
    data = s.encode("utf-8")
    buf += struct.pack(">H", len(data))
    buf += data


def _write_payload(buf: bytearray, tag: int, value: Any) -> None:
    if tag == TAG_BYTE:
        buf += struct.pack(">b", int(value))
    elif tag == TAG_SHORT:
        buf += struct.pack(">h", int(value))
    elif tag == TAG_INT:
        buf += struct.pack(">i", int(value))
    elif tag == TAG_LONG:
        buf += struct.pack(">q", int(value))
    elif tag == TAG_FLOAT:
        buf += struct.pack(">f", float(value))
    elif tag == TAG_DOUBLE:
        buf += struct.pack(">d", float(value))
    elif tag == TAG_BYTE_ARRAY:
        data = bytes(int(b) & 0xFF for b in value)
        buf += struct.pack(">i", len(data))
        buf += data
    elif tag == TAG_STRING:
        _write_string(buf, str(value))
    elif tag == TAG_LIST:
        item_type = value.item_type
        buf += struct.pack(">bi", item_type, len(value))
        for item in value:
            _write_payload(buf, item_type, item)
    elif tag == TAG_COMPOUND:
        for key, val in value.items():
            sub_tag = _infer_tag(val)
            buf += struct.pack(">b", sub_tag)
            _write_string(buf, key)
            _write_payload(buf, sub_tag, val)
        buf += struct.pack(">b", TAG_END)
    elif tag == TAG_INT_ARRAY:
        buf += struct.pack(">i", len(value))
        for v in value:
            buf += struct.pack(">i", _i32(int(v)))
    elif tag == TAG_LONG_ARRAY:
        buf += struct.pack(">i", len(value))
        for v in value:
            buf += struct.pack(">q", _i64(int(v)))
    else:
        raise ValueError(f"不支持的 NBT 标签类型: {tag}")


def _infer_tag(value: Any) -> int:
    if isinstance(value, IntArray):
        return TAG_INT_ARRAY
    if isinstance(value, LongArray):
        return TAG_LONG_ARRAY
    if isinstance(value, ListTag):
        return TAG_LIST
    if isinstance(value, dict):
        return TAG_COMPOUND
    if isinstance(value, bool):
        return TAG_BYTE
    if isinstance(value, Long):
        return TAG_LONG
    if isinstance(value, int):
        return TAG_INT
    if isinstance(value, float):
        return TAG_DOUBLE
    if isinstance(value, str):
        return TAG_STRING
    if isinstance(value, (bytes, bytearray)):
        return TAG_BYTE_ARRAY
    raise TypeError(f"无法推断 NBT 标签类型: {type(value)!r}")


def write_nbt(root: dict) -> bytes:
    """把 dict 写成 NBT 字节流（根标签名为空字符串）。"""
    buf = bytearray()
    buf += struct.pack(">b", TAG_COMPOUND)
    _write_string(buf, "")
    _write_payload(buf, TAG_COMPOUND, root)
    return bytes(buf)


# ---------------------------------------------------------------------------
# 读取（用于自测与校验）
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("NBT 数据不足")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def u8(self) -> int:
        return struct.unpack(">B", self.read(1))[0]

    def i8(self) -> int:
        return struct.unpack(">b", self.read(1))[0]

    def i16(self) -> int:
        return struct.unpack(">h", self.read(2))[0]

    def u16(self) -> int:
        return struct.unpack(">H", self.read(2))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack(">q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack(">f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack(">d", self.read(8))[0]

    def string(self) -> str:
        n = struct.unpack(">H", self.read(2))[0]
        return self.read(n).decode("utf-8", "replace")


def _read_payload(r: _Reader, tag: int) -> Any:
    if tag == TAG_BYTE:
        return r.i8()
    if tag == TAG_SHORT:
        return r.i16()
    if tag == TAG_INT:
        return r.i32()
    if tag == TAG_LONG:
        return r.i64()
    if tag == TAG_FLOAT:
        return r.f32()
    if tag == TAG_DOUBLE:
        return r.f64()
    if tag == TAG_BYTE_ARRAY:
        n = r.i32()
        return list(r.read(n))
    if tag == TAG_STRING:
        return r.string()
    if tag == TAG_LIST:
        item_type = r.u8()
        n = r.i32()
        return ListTag(item_type, [_read_payload(r, item_type) for _ in range(n)])
    if tag == TAG_COMPOUND:
        out: dict[str, Any] = {}
        while True:
            sub = r.u8()
            if sub == TAG_END:
                break
            key = r.string()
            out[key] = _read_payload(r, sub)
        return out
    if tag == TAG_INT_ARRAY:
        n = r.i32()
        return IntArray(r.i32() for _ in range(n))
    if tag == TAG_LONG_ARRAY:
        n = r.i32()
        return LongArray(r.i64() for _ in range(n))
    raise ValueError(f"不支持的 NBT 标签类型: {tag}")


def read_nbt(data: bytes) -> dict:
    """读回 NBT，返回根 Compound（不含根标签名）。"""
    r = _Reader(data)
    tag = r.u8()
    if tag != TAG_COMPOUND:
        raise ValueError(f"根标签不是 Compound: {tag}")
    r.string()
    return _read_payload(r, TAG_COMPOUND)


def gzip_bytes(data: bytes) -> bytes:
    """gzip 压缩。litematic 要求 gzip 格式。"""
    return gzip.compress(data, compresslevel=6)
