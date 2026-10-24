"""音高转 MIDI 音符：把音频转成 (start, duration, midi, velocity) 事件序列。

主路径是 Spotify basic-pitch 的 ONNX 模型（nmp.onnx），后处理严格对齐官方
`basic_pitch/note_creation.py` 的 `output_to_notes_polyphonic` 算法（含 melodia trick）。
若模型不可用，退回自研的 FFT 峰值 + 谐波匹配单基频方案。

basic-pitch 关键常量（见 basic_pitch/constants.py）：
    AUDIO_SAMPLE_RATE = 22050
    FFT_HOP          = 256
    AUDIO_WINDOW_LENGTH = 2 秒
    AUDIO_N_SAMPLES  = 22050 * 2 - 256 = 43844
    ANNOTATIONS_FPS  = 22050 // 256 = 86
    ANNOT_N_FRAMES   = 86 * 2 = 172
    ANNOTATIONS_BASE_FREQUENCY = 27.5 (A0)
    MIDI_OFFSET      = 21
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .paths import models_dir

MODEL_DIR = str(models_dir())
NMP_MODEL = os.path.join(MODEL_DIR, "nmp.onnx")

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
AUDIO_WINDOW_LENGTH = 2
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP  # 86
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH  # 172
N_OVERLAPPING_FRAMES = 30
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
ENERGY_TOLERANCE = 11
MAGIC_ALIGNMENT_OFFSET = 0.0018

DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 127.7


@dataclass
class NoteEvent:
    """一个音符事件。"""

    start: float  # 秒
    duration: float  # 秒
    midi: int  # 0..127
    velocity: float  # 0..1

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def frequency(self) -> float:
        return 440.0 * (2.0 ** ((self.midi - 69) / 12.0))


@dataclass
class Transcription:
    notes: list[NoteEvent] = field(default_factory=list)
    source: str = "none"

    def stats(self) -> dict:
        if not self.notes:
            return {"count": 0, "source": self.source}
        m = [n.midi for n in self.notes]
        return {
            "count": len(self.notes),
            "source": self.source,
            "min_midi": min(m),
            "max_midi": max(m),
            "avg_duration": round(sum(n.duration for n in self.notes) / len(self.notes), 3),
        }


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def midi_to_freq(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def freq_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * math.log2(max(freq, 1e-6) / 440.0)


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性重采样。"""
    if src_rate == dst_rate:
        return samples
    n_src = samples.shape[-1]
    n_dst = int(round(n_src * dst_rate / src_rate))
    if n_dst <= 0 or n_src <= 0:
        return np.zeros_like(samples, dtype=np.float32)
    src_idx = np.linspace(0.0, n_src - 1.0, num=n_dst, dtype=np.float64)
    return np.interp(src_idx, np.arange(n_src), samples).astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX 模型封装
# ---------------------------------------------------------------------------

class BasicPitchOnnx:
    """basic-pitch nmp 模型的薄封装。"""

    # basic-pitch 官方对 ONNX 输出的取名顺序：note / onset / contour
    OUT_NOTE = "StatefulPartitionedCall:1"
    OUT_ONSET = "StatefulPartitionedCall:2"
    OUT_CONTOUR = "StatefulPartitionedCall:0"

    def __init__(self, model_path: str = NMP_MODEL):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """window: (n, 43844, 1) → (note, onset, contour)。

        note/onset 形状 (n, 172, 88)，contour 形状 (n, 172, 264)。
        """
        outs = self.session.run(
            [self.OUT_NOTE, self.OUT_ONSET, self.OUT_CONTOUR],
            {self.input_name: window.astype(np.float32)},
        )
        return outs[0], outs[1], outs[2]


# ---------------------------------------------------------------------------
# 官方算法移植
# ---------------------------------------------------------------------------

def _get_infered_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """从帧幅度跳变推断 onset（官方 get_infered_onsets）。"""
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    max_diff = float(frame_diff.max())
    if max_diff > 0:
        frame_diff = np.max(onsets) * frame_diff / max_diff

    return np.max([onsets, frame_diff], axis=0)


def _constrain_frequency(
    onsets: np.ndarray, frames: np.ndarray, max_freq: float | None, min_freq: float | None
) -> tuple[np.ndarray, np.ndarray]:
    n_freqs = onsets.shape[1]
    min_freq_idx = 0
    max_freq_idx = n_freqs
    if min_freq is not None:
        min_freq_idx = int(np.round(freq_to_midi(min_freq) - MIDI_OFFSET))
    if max_freq is not None:
        max_freq_idx = int(np.round(freq_to_midi(max_freq) - MIDI_OFFSET))
    min_freq_idx = max(0, min(min_freq_idx, n_freqs))
    max_freq_idx = max(0, min(max_freq_idx, n_freqs))
    onsets[:, :min_freq_idx] = 0
    frames[:, :min_freq_idx] = 0
    onsets[:, max_freq_idx:] = 0
    frames[:, max_freq_idx:] = 0
    return onsets, frames


def _argrelmax(a: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """等价于 scipy.signal.argrelmax（去掉 scipy 依赖）。"""
    if a.shape[axis] < 3:
        empty = np.array([], dtype=int)
        return empty, empty
    a = np.moveaxis(a, axis, 0)
    mask = (a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])
    idx = np.nonzero(mask)[0] + 1
    return idx, np.nonzero(mask)[1]


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool = True,
    max_freq: float | None = None,
    min_freq: float | None = None,
    melodia_trick: bool = True,
    energy_tol: int = ENERGY_TOLERANCE,
) -> list[tuple[int, int, int, float]]:
    """官方 output_to_notes_polyphonic 的移植。返回 (start_frame, end_frame, midi, amp)。"""
    n_frames = frames.shape[0]
    if n_frames == 0:
        return []

    onsets = onsets.copy()
    frames = frames.copy()
    onsets, frames = _constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = _get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = _argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = frames.copy()
    note_events: list[tuple[int, int, int, float]] = []

    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue

        i = note_start_idx + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k

        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        amplitude = float(np.mean(frames[note_start_idx:i, freq_idx]))
        note_events.append((note_start_idx, i, freq_idx + MIDI_OFFSET, amplitude))

    if melodia_trick:
        energy_shape = remaining_energy.shape
        while remaining_energy.size and float(remaining_energy.max()) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(int(np.argmax(remaining_energy)), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k

            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k

            if i_end <= i_start or i_end - i_start <= min_note_len:
                continue

            amplitude = float(np.mean(frames[i_start:i_end, freq_idx]))
            note_events.append((i_start, i_end, freq_idx + MIDI_OFFSET, amplitude))

    return note_events


def model_frames_to_time(n_frames: int) -> np.ndarray:
    """官方 model_frames_to_time 的移植（不依赖 librosa）。"""
    original_times = np.arange(n_frames) * (FFT_HOP / AUDIO_SAMPLE_RATE)
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + MAGIC_ALIGNMENT_OFFSET
    return original_times - (window_offset * window_numbers)


# ---------------------------------------------------------------------------
# 推理主流程
# ---------------------------------------------------------------------------

def _window_audio(audio: np.ndarray, hop_size: int):
    """官方 window_audio_file：按 hop 切窗，不足则补零。"""
    for i in range(0, audio.shape[0], hop_size):
        window = audio[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, pad_width=[[0, AUDIO_N_SAMPLES - len(window)]])
        yield np.expand_dims(window, axis=-1)


def _unwrap_output(output: np.ndarray, audio_original_length: int, hop_size: int) -> np.ndarray:
    """官方 unwrap_output 的移植。"""
    if output.ndim != 3:
        return np.zeros((0, 0))
    n_olap = int(0.5 * N_OVERLAPPING_FRAMES)
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]
    shape = output.shape
    unwrapped = output.reshape(shape[0] * shape[1], shape[2])
    n_expected_windows = audio_original_length / hop_size
    n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - N_OVERLAPPING_FRAMES
    return unwrapped[: int(n_expected_windows * n_frames_per_window), :]


def run_model(
    samples: np.ndarray,
    sr: int,
    model: BasicPitchOnnx,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跑模型，返回展开后的 (note, onset, contour) 矩阵。"""
    audio = resample_linear(samples, sr, AUDIO_SAMPLE_RATE)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    original_length = audio.shape[0]
    overlap_len = N_OVERLAPPING_FRAMES * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio])

    notes_out: list[np.ndarray] = []
    onsets_out: list[np.ndarray] = []
    contours_out: list[np.ndarray] = []

    windows = list(_window_audio(audio, hop_size))
    for idx, window in enumerate(windows):
        n, o, c = model.predict(window[None, :, :])
        notes_out.append(n)
        onsets_out.append(o)
        contours_out.append(c)
        if progress is not None and len(windows) > 1:
            progress((idx + 1) / len(windows))

    note = _unwrap_output(np.concatenate(notes_out), original_length, hop_size)
    onset = _unwrap_output(np.concatenate(onsets_out), original_length, hop_size)
    contour = _unwrap_output(np.concatenate(contours_out), original_length, hop_size)
    return note, onset, contour


def transcribe_basic_pitch(
    samples: np.ndarray,
    sr: int,
    model: BasicPitchOnnx | None = None,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    min_freq: float | None = None,
    max_freq: float | None = None,
    melodia_trick: bool = True,
    progress=None,
) -> Transcription:
    """用 basic-pitch 转录音频。samples 为单声道 float32。"""
    if model is None:
        model = get_basic_pitch_model()
    if model is None:
        raise RuntimeError("basic-pitch 模型不可用")

    note, onset, _contour = run_model(samples, sr, model, progress=progress)
    if note.size == 0:
        return Transcription(source="basic-pitch")

    min_note_len = int(
        round(minimum_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP))
    )
    events = output_to_notes_polyphonic(
        note,
        onset,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        infer_onsets=True,
        min_freq=min_freq,
        max_freq=max_freq,
        melodia_trick=melodia_trick,
    )

    times = model_frames_to_time(note.shape[0])
    notes: list[NoteEvent] = []
    for start_idx, end_idx, midi, amp in events:
        t0 = float(times[start_idx]) if start_idx < len(times) else 0.0
        t1 = float(times[end_idx]) if end_idx < len(times) else t0
        notes.append(
            NoteEvent(start=max(0.0, t0), duration=max(0.01, t1 - t0), midi=int(midi), velocity=float(amp))
        )

    notes.sort(key=lambda n: (n.start, n.midi))
    return Transcription(notes=notes, source="basic-pitch")


# ---------------------------------------------------------------------------
# 备选路径：自研单基频检测
# ---------------------------------------------------------------------------

def _windowed_spectrogram(samples: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    """返回 (frames, n_fft//2+1) 的幅度谱。samples 必须是 1-D 单声道。"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=0).astype(np.float32)
    samples = np.ascontiguousarray(samples)
    if samples.size == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))
    n_frames = 1 + (samples.size - n_fft) // hop
    # 用显式索引构造帧矩阵（as_strided 在非连续输入上会产生非法视图）
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[idx] * win
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).astype(np.float32)


def transcribe_spectral(
    samples: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    hop: int = 512,
    min_freq: float = 55.0,
    max_freq: float = 2000.0,
    peak_ratio: float = 0.28,
    min_duration: float = 0.12,
) -> Transcription:
    """无模型时的退路：谱峰 + 谐波一致性检测单音。"""
    if samples.size == 0:
        return Transcription(source="spectral")

    spec = _windowed_spectrogram(samples, sr, n_fft, hop)
    if spec.size == 0:
        return Transcription(source="spectral")

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= min_freq) & (freqs <= max_freq)
    band_idx = np.where(band)[0]
    if band_idx.size == 0:
        return Transcription(source="spectral")

    sub = spec[:, band_idx]
    band_freqs = freqs[band_idx]

    notes: list[NoteEvent] = []
    cur: NoteEvent | None = None
    cur_hits = 0

    for fi in range(sub.shape[0]):
        frame = sub[fi]
        fmax = float(frame.max())
        if fmax <= 1e-9:
            if cur is not None and cur_hits >= 3:
                cur.duration = fi * hop / sr - cur.start
                if cur.duration >= min_duration:
                    notes.append(cur)
            cur = None
            cur_hits = 0
            continue

        cand = np.where((frame[1:-1] > frame[:-2]) & (frame[1:-1] >= frame[2:]))[0] + 1
        if cand.size == 0:
            cand = np.array([int(np.argmax(frame))])
        cand = cand[frame[cand] >= fmax * peak_ratio]
        if cand.size == 0:
            continue

        best = None
        best_score = 0.0
        for ci in cand:
            f0 = band_freqs[ci]
            if f0 < min_freq:
                continue
            score = 0.0
            for h in range(1, 5):
                target = f0 * h
                if target > sr / 2:
                    break
                ti = int(np.argmin(np.abs(band_freqs - target)))
                if abs(band_freqs[ti] - target) > max(20.0, target * 0.03):
                    continue
                score += float(frame[ti]) / fmax
            if score > best_score:
                best_score = score
                best = f0

        if best is None:
            continue

        midi = int(np.clip(round(freq_to_midi(best)), 12, 108))
        if cur is not None and cur.midi == midi:
            cur_hits += 1
        else:
            if cur is not None and cur_hits >= 3 and cur.duration >= min_duration:
                notes.append(cur)
            cur = NoteEvent(start=fi * hop / sr, duration=0.0, midi=midi, velocity=0.7)
            cur_hits = 1

    if cur is not None and cur_hits >= 3:
        cur.duration = sub.shape[0] * hop / sr - cur.start
        if cur.duration >= min_duration:
            notes.append(cur)

    return Transcription(notes=notes, source="spectral")


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, object] = {}


def get_basic_pitch_model() -> BasicPitchOnnx | None:
    """加载（并缓存）basic-pitch 模型；文件缺失或加载失败返回 None。"""
    if "bp" in _MODEL_CACHE:
        return _MODEL_CACHE["bp"]  # type: ignore[return-value]
    if not os.path.isfile(NMP_MODEL):
        _MODEL_CACHE["bp"] = None
        return None
    try:
        model = BasicPitchOnnx()
    except Exception:
        _MODEL_CACHE["bp"] = None
        return None
    _MODEL_CACHE["bp"] = model
    return model


def transcribe(
    samples: np.ndarray,
    sr: int,
    prefer_model: bool = True,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    melodia_trick: bool = True,
    progress=None,
) -> Transcription:
    """转录音频为音符事件。"""
    if samples.size == 0:
        return Transcription(source="empty")
    # 模型与频谱路径都只接受单声道；多声道先下混
    if samples.ndim > 1:
        samples = samples.mean(axis=0).astype(np.float32)
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    # 静音 / 极小声检测：直接返回空，避免模型把数值噪声当成音符
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak < 1e-3:
        return Transcription(source="silent")
    if prefer_model:
        model = get_basic_pitch_model()
        if model is not None:
            try:
                res = transcribe_basic_pitch(
                    samples,
                    sr,
                    model,
                    onset_threshold=onset_threshold,
                    frame_threshold=frame_threshold,
                    minimum_note_length_ms=minimum_note_length_ms,
                    melodia_trick=melodia_trick,
                    progress=progress,
                )
                if res.notes:
                    return res
                print("[转录] 模型未检出音符，退回频谱法")
            except Exception as exc:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"[转录] basic-pitch 模型推理失败（{exc}），退回频谱法")
    return transcribe_spectral(samples, sr)
