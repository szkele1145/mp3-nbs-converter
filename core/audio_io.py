"""音频读取：统一解码为单声道/立体声 float32 数组。

优先用 ffmpeg（系统里已装）解码 mp3，ffmpeg 不可用时退回 librosa/audioread。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 44100


@dataclass
class Audio:
    """解码后的音频。samples 形状为 (channels, n)。"""

    samples: np.ndarray  # float32, (channels, n)
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sample_rate

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=0).astype(np.float32)

    def peak_normalize(self, target: float = 0.98) -> None:
        peak = float(np.max(np.abs(self.samples))) if self.samples.size else 0.0
        if peak > 1e-9:
            self.samples = (self.samples * (target / peak)).astype(np.float32)


def ffmpeg_binary() -> str | None:
    """定位 ffmpeg 可执行文件。"""
    for name in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(name)
        if found:
            return found
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "ffmpeg.exe")
    if os.path.isfile(local):
        return local
    return None


def decode(path: str, sample_rate: int = SAMPLE_RATE, mono: bool = False) -> Audio:
    """解码任意 ffmpeg 支持的音频文件为 float32 数组。"""
    exe = ffmpeg_binary()
    if exe is None:
        return _decode_librosa(path, sample_rate, mono)
    return _decode_ffmpeg(exe, path, sample_rate, mono)


def _decode_ffmpeg(exe: str, path: str, sample_rate: int, mono: bool) -> Audio:
    channels = 1 if mono else 2
    cmd = [
        exe, "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            "ffmpeg 解码失败: " + proc.stderr.decode("utf-8", "replace")[-500:]
        )
    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels == 1:
        data = data[None, :]
    else:
        data = data.reshape(-1, channels).T
    return Audio(samples=np.ascontiguousarray(data, dtype=np.float32), sample_rate=sample_rate)


def _decode_librosa(path: str, sample_rate: int, mono: bool) -> Audio:
    import librosa

    data, sr = librosa.load(path, sr=sample_rate, mono=mono)
    data = np.asarray(data, dtype=np.float32)
    if mono:
        data = data[None, :]
    else:
        data = data[None, :]
    return Audio(samples=data, sample_rate=sample_rate)


def write_wav(path: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """把 (channels, n) 或 (n,) 的 float 数组写成 16bit WAV。"""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr.T * 32767.0).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(arr.shape[0])
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def resample_to(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性重采样（够用且快；高质量场景用 librosa.resample）。"""
    if src_rate == dst_rate:
        return samples
    n_src = samples.shape[-1]
    n_dst = int(round(n_src * dst_rate / src_rate))
    if n_dst <= 0 or n_src <= 0:
        return np.zeros((samples.shape[0], n_dst), dtype=np.float32)
    src_idx = np.linspace(0.0, n_src - 1.0, num=n_dst, dtype=np.float64)
    if samples.ndim == 1:
        return np.interp(src_idx, np.arange(n_src), samples).astype(np.float32)
    out = np.empty((samples.shape[0], n_dst), dtype=np.float32)
    for ch in range(samples.shape[0]):
        out[ch] = np.interp(src_idx, np.arange(n_src), samples[ch]).astype(np.float32)
    return out


def to_temp_wav(audio: Audio) -> str:
    """把内存中的音频写成临时 wav，返回路径。"""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(path, audio.samples, audio.sample_rate)
    return path
