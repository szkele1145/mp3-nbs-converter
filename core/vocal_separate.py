"""浜哄０鍒嗙锛堟櫤鑳藉幓浜哄０锛屽彧鐣欎即濂忥級"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort

from .config import low_memory, threads as _threads
from .paths import models_dir

MODEL_DIR = Path(models_dir())
SEG = 343980
N_FFT = 4096
HOP = 1024
BINS = 2048
N_FRAMES = 336

MODELS = [
    ("htdemucs_ft_vocals", "htdemucs_ft_vocals.onnx", "HTDemucs-FT"),
    ("htdemucs", "htdemucs_onnx.onnx", "HTDemucs"),
    ("UVR_MDXNET_Main", "UVR_MDXNET_Main.onnx", "UVR MDX-NET"),
]

_session_cache: dict[str, object] = {}

#: 最近一次 separate_vocals 是否退化为频谱法（模型不可用/推理失败）
LAST_WAS_FALLBACK: bool = False


def list_models() -> list[dict]:
    out = []
    for mid, fname, name in MODELS:
        p = MODEL_DIR / fname
        out.append({"id": mid, "name": name, "file": fname, "available": p.exists()})
    return out


def is_available() -> bool:
    return any((MODEL_DIR / f).exists() for _, f, _ in MODELS)


def clear_model_cache() -> None:
    _session_cache.clear()


def _load_session(model_id: str):
    key = model_id
    if key in _session_cache:
        return _session_cache[key]
    fname = None
    for mid, f, _ in MODELS:
        if mid == model_id:
            fname = f
            break
    if fname is None:
        fname = f"{model_id}.onnx"
    path = MODEL_DIR / fname
    if not path.exists():
        # 找出可用的模型，给出友好提示
        avail = [mid for mid, f, _ in MODELS if (MODEL_DIR / f).exists()]
        if avail:
            raise FileNotFoundError(
                f"模型文件缺失：{path}。"
                f"当前可用的模型：{', '.join(avail)}。"
                f"也可以从 UVR/Demucs 官方下载 .onnx 放到 models/ 目录。"
            )
        raise FileNotFoundError(
            f"找不到任何模型文件（models/ 目录为空）。"
            f"请把 .onnx 模型放到 {MODEL_DIR} 后重试。"
        )
    opts = ort.SessionOptions()
    t = _threads()
    opts.intra_op_num_threads = max(1, int(t))
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    _session_cache[key] = sess
    return sess


def _hann(n: int) -> np.ndarray:
    return np.hanning(n).astype(np.float32)


def _transition_window(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    return np.sqrt(np.maximum(w, 1e-12))


def _stft(x: np.ndarray):
    win = _hann(N_FFT)
    pad = N_FFT // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(n_frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
    frames = xp[idx] * win[None, :]
    spec = np.fft.rfft(frames, axis=-1)
    return spec.T.astype(np.complex64)


def _istft(spec: np.ndarray, length: int):
    win = _hann(N_FFT)
    frames = np.fft.irfft(spec.T, n=N_FFT)
    out = np.zeros(length + N_FFT, dtype=np.float32)
    count = np.zeros(length + N_FFT, dtype=np.float32)
    for i in range(frames.shape[0]):
        s = i * HOP
        out[s : s + N_FFT] += frames[i] * win
        count[s : s + N_FFT] += win
    count = np.maximum(count, 1e-8)
    out = out[:length] / count[:length]
    return out


def _demucs_build_inputs(chunk: np.ndarray):
    le = int(np.ceil(chunk.shape[1] / HOP) * HOP) + N_FFT
    if chunk.shape[1] < le:
        pad = le - chunk.shape[1]
        chunk = np.pad(chunk, ((0, 0), (0, pad)), mode="constant")
    x = np.zeros((1, 2, 4, BINS, N_FRAMES), dtype=np.float32)
    for i in range(2):
        X = _stft(chunk[i])
        real = X.real.astype(np.float32)
        imag = X.imag.astype(np.float32)
        mag = np.sqrt(real * real + imag * imag + 1e-8)
        x[0, i, 0] = real[:BINS, :N_FRAMES]
        x[0, i, 1] = imag[:BINS, :N_FRAMES]
        x[0, i, 2] = mag[:BINS, :N_FRAMES]
        x[0, i, 3] = np.angle(X)[:BINS, :N_FRAMES]
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), keepdims=True)
    x = ((x - mean) / (1e-5 + std)).astype(np.float32)
    return chunk[None].astype(np.float32), x


DRUMS_INDEX = 0
BASS_INDEX = 1
OTHER_INDEX = 2
VOCALS_INDEX = 3


def separate_htdemucs(samples: np.ndarray, sr: int, model_id: str = "htdemucs_ft_vocals", progress=None):
    sess = _load_session(model_id)
    ch, n = samples.shape
    if ch == 1:
        stereo_full = np.repeat(samples, 2, axis=0)
    else:
        stereo_full = samples[:2]

    peak = float(np.abs(stereo_full).max())
    scale = 1.0 / peak if peak > 1e-6 else 1.0
    stereo_full = (stereo_full * scale).astype(np.float32)

    stems = np.zeros((4, 2, n), dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    stride = SEG - SEG // 4
    window = _transition_window(SEG)
    n_chunks = max(1, int(np.ceil(n / stride)))

    # 输入/输出按模型实际声明来，不要写死：
    # 现有导出版本都是单输入 'mix' → (1, 4, 2, SEG) 的 stems。
    in_names = [i.name for i in sess.get_inputs()]
    feed_two = len(in_names) >= 2
    out_names = [o.name for o in sess.get_outputs()]

    for i in range(n_chunks):
        start = i * stride
        end = min(start + SEG, n)
        chunk = stereo_full[:, start:end]
        chunk_len = chunk.shape[1]
        if chunk_len < SEG:
            chunk = np.pad(chunk, ((0, 0), (0, SEG - chunk_len)), mode="constant")
        if feed_two:
            inp, x = _demucs_build_inputs(chunk)
            feed = {in_names[0]: inp, in_names[1]: x}
        else:
            # 单输入模型：直接喂波形 (1, 2, SEG)
            feed = {in_names[0]: chunk[None].astype(np.float32)}
        outs = sess.run(None, feed)

        # 找时间域的 4 源输出 (1, 4, 2, SEG)；有的导出是 (1,4,2,SEG)，
        # 也有的是 (4,2,SEG) 或双输出（频谱 + 波形），这里按形状挑。
        time_out = None
        for o in outs:
            arr = np.asarray(o)
            if arr.ndim == 4 and arr.shape[1] == 4:
                time_out = arr[0]
                break
            if arr.ndim == 3 and arr.shape[0] == 4:
                time_out = arr
                break
        if time_out is None:
            raise ValueError(
                f"模型输出形状不认识：{[np.asarray(o).shape for o in outs]}"
            )
        if time_out.shape[-1] < chunk_len:
            time_out = np.pad(
                time_out, ((0, 0), (0, 0), (0, chunk_len - time_out.shape[-1]))
            )
        w = window[:chunk_len]
        for s in range(4):
            stems[s, :, start:end] += time_out[s, :, :chunk_len] * w
        weight[start:end] += w
        if progress is not None:
            try:
                progress((i + 1) / n_chunks)
            except Exception:  # noqa: BLE001
                pass

    weight = np.maximum(weight, 1e-8)
    stems /= weight[None, None, :]
    stems /= scale

    is_specialist = "ft" in model_id.lower()
    if is_specialist:
        vocals = stems[VOCALS_INDEX]
        accomp = stereo_full - vocals
    else:
        vocals = stems[VOCALS_INDEX]
        accomp = stems[DRUMS_INDEX] + stems[BASS_INDEX] + stems[OTHER_INDEX]

    if ch == 1:
        accomp, vocals = accomp[:1], vocals[:1]
    elif ch > 2:
        extra = ch - 2
        accomp = np.concatenate([accomp, np.repeat(accomp[-1:], extra, axis=0)], axis=0)
        vocals = np.concatenate([vocals, np.repeat(vocals[-1:], extra, axis=0)], axis=0)

    if low_memory():
        clear_model_cache()

    return accomp.astype(np.float32), vocals.astype(np.float32)


# UVR MDX-NET
MDX_FREQ_BINS = 3072
MDX_TIME_FRAMES = 256
MDX_N_FFT = 6144
MDX_HOP = 1024


def _mdx_hann(n_fft: int) -> np.ndarray:
    return np.hanning(n_fft + 1)[:-1].astype(np.float32)


def _mdx_stft(x: np.ndarray, n_fft: int = MDX_N_FFT, hop: int = MDX_HOP) -> np.ndarray:
    pad = n_fft // 2
    xp = np.pad(np.asarray(x, dtype=np.float32), (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    win = _mdx_hann(n_fft)
    frames = np.stack([xp[i * hop : i * hop + n_fft] for i in range(n_frames)])
    frames = frames * win
    return np.fft.rfft(frames, axis=-1).T.astype(np.complex64)


def _mdx_istft(spec: np.ndarray, length: int, n_fft: int = MDX_N_FFT, hop: int = MDX_HOP) -> np.ndarray:
    win = _mdx_hann(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).T.astype(np.float32)
    frames = frames * win
    n_frames = frames.shape[0]
    total = n_fft + hop * (n_frames - 1)
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    w2 = win ** 2
    for i in range(n_frames):
        out[i * hop : i * hop + n_fft] += frames[i]
        norm[i * hop : i * hop + n_fft] += w2
    norm[norm < 1e-8] = 1e-8
    out /= norm
    pad = n_fft // 2
    out = out[pad : pad + length]
    if len(out) < length:
        out = np.pad(out, (0, length - len(out)))
    return out.astype(np.float32)


def separate_mdxnet(samples: np.ndarray, sr: int, model_id: str = "UVR_MDXNET_Main", progress=None):
    sess = _load_session(model_id)
    ch, n = samples.shape
    if ch == 1:
        stereo = np.repeat(samples, 2, axis=0)
    else:
        stereo = samples[:2].astype(np.float32)

    peak = float(np.abs(stereo).max())
    scale = 1.0 / peak if peak > 1e-6 else 1.0
    stereo = (stereo * scale).astype(np.float32)

    S = np.stack([_mdx_stft(stereo[0]), _mdx_stft(stereo[1])])[:, :MDX_FREQ_BINS]
    total = S.shape[2]
    dim_t = MDX_TIME_FRAMES
    dim_f = MDX_FREQ_BINS
    step = (dim_t * 3) // 4  # 25% overlap

    voc_spec = np.zeros((2, dim_f, total), dtype=np.complex64)
    weight = np.zeros(total, dtype=np.float32)
    inp_name = sess.get_inputs()[0].name
    n_windows = max(1, int(np.ceil(total / step)))

    # 注意：模型第一维虽是动态 batch，但实测批处理对吞吐没有提升
    # （推理是计算密集型，batch 越大单次越慢，总时间不变，内存还翻倍），
    # 所以这里保持逐窗口推理，内存占用最低。
    for wi, f0 in enumerate(range(0, total, step)):
        f1 = min(f0 + dim_t, total)
        take = f1 - f0
        seg = S[:, :, f0:f1]
        if take < dim_t:
            seg = np.pad(seg, ((0, 0), (0, 0), (0, dim_t - take)))
        inp = np.stack([seg[0].real, seg[0].imag, seg[1].real, seg[1].imag], axis=0)[None].astype(np.float32)
        out = sess.run(None, {inp_name: inp})[0][0]
        voc_spec[0, :, f0:f1] += (out[0] + 1j * out[1])[:, :take]
        voc_spec[1, :, f0:f1] += (out[2] + 1j * out[3])[:, :take]
        weight[f0:f1] += 1.0
        if progress is not None:
            try:
                progress((wi + 1) / n_windows)
            except Exception:  # noqa: BLE001
                pass

    weight = np.maximum(weight, 1e-8)
    voc_spec = voc_spec / weight[None, None, :]

    def _to_wave(Z: np.ndarray) -> np.ndarray:
        full = np.zeros((MDX_N_FFT // 2 + 1, Z.shape[1]), dtype=np.complex64)
        full[:dim_f] = Z
        return _mdx_istft(full, n)

    voc = np.stack([_to_wave(voc_spec[0]), _to_wave(voc_spec[1])])
    orig = samples[:2].astype(np.float32) if ch >= 2 else np.repeat(samples[:1].astype(np.float32), 2, axis=0)
    voc = voc / scale
    accomp = orig - voc
    if low_memory():
        clear_model_cache()
    if ch == 1:
        return accomp[:1], voc[:1]
    if ch > 2:
        extra = ch - 2
        accomp = np.concatenate([accomp, np.repeat(accomp[-1:], extra, axis=0)], axis=0)
        voc = np.concatenate([voc, np.repeat(voc[-1:], extra, axis=0)], axis=0)
    return accomp.astype(np.float32), voc.astype(np.float32)


def separate_spectral(samples: np.ndarray, sr: int):
    ch, n = samples.shape
    if ch == 1:
        stereo = np.repeat(samples, 2, axis=0)
    else:
        stereo = samples[:2]
    L = _stft(stereo[0])
    R = _stft(stereo[1])
    mag = np.sqrt(np.abs(L) ** 2 + np.abs(R) ** 2 + 1e-12)
    phase = (L + 1j * R) / (mag + 1e-12)
    mask = np.ones_like(mag)
    f = np.linspace(0, sr / 2, mag.shape[0])
    mid = (f > 200) & (f < 4000)
    mask[mid] *= 0.25
    acc_spec = phase * mag * mask
    acc = np.stack([_istft(acc_spec.real, n), _istft(acc_spec.imag, n)])
    if ch == 1:
        return acc[:1], stereo[:1] - acc[:1]
    if ch > 2:
        extra = ch - 2
        acc = np.concatenate([acc, np.repeat(acc[-1:], extra, axis=0)], axis=0)
    return acc.astype(np.float32), (stereo - acc).astype(np.float32)


def separate_vocals(samples: np.ndarray, sr: int, model: str = "htdemucs_ft_vocals", progress=None) -> Tuple[np.ndarray, np.ndarray]:
    """按模型名分发到具体实现。

    模型不可用时退化为频谱中频抑制法，保证功能不中断；但会打印明确警告，
    避免用户误以为「已经去干净人声了」。

    退化时会把模块级标志 ``LAST_WAS_FALLBACK`` 置为 True，
    调用方据此决定是否把结果写进缓存。
    """
    global LAST_WAS_FALLBACK
    LAST_WAS_FALLBACK = False
    model_l = (model or "").lower()
    try:
        if "uvr" in model_l or "mdx" in model_l:
            return separate_mdxnet(samples, sr, model, progress=progress)
        if model_l in ("spectral", "none", "off"):
            LAST_WAS_FALLBACK = True
            return separate_spectral(samples, sr)
        return separate_htdemucs(samples, sr, model, progress=progress)
    except FileNotFoundError:
        # 模型文件缺失：明确告知，退化为频谱法
        print(f"[去人声] 模型 {model} 不可用，退化为频谱法（去人声效果会明显变差）")
        LAST_WAS_FALLBACK = True
        return separate_spectral(samples, sr)
    except Exception as exc:  # noqa: BLE001
        print(f"[去人声] 模型 {model} 推理失败（{type(exc).__name__}: {exc}），退化为频谱法")
        LAST_WAS_FALLBACK = True
        try:
            return separate_spectral(samples, sr)
        except Exception:
            raise


def apply_to_stereo(samples: np.ndarray):
    if samples.ndim == 1:
        return np.stack([samples, samples])
    if samples.shape[0] == 1:
        return np.repeat(samples, 2, axis=0)
    return samples[:2]
