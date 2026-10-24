"""转换流水线：音频 → 音符 → .nbs / .midi / .litematic。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import compact_layout as cl
from . import nbs_writer, midi_writer
from .audio_io import Audio, decode, write_wav

GAME_TICKS_PER_SEC = 20.0   # Minecraft 游戏刻
NBS_TICKS_PER_SEC = 10.0    # NBS 红石刻


@dataclass
class ConvertOptions:
    """转换参数。"""

    remove_vocals: bool = True
    vocal_model: str = "htdemucs_ft_vocals"
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_ms: float = 127.7
    melodia_trick: bool = True
    instrument: str = "HARP"
    bass_instrument: str = "DOUBLE_BASS"
    coarse: int = 0
    wrap: int = 0
    gap: int = 0
    max_seconds: float = 0.0
    bass_split_midi: int = 55
    keep_nbs: bool = True
    keep_midi: bool = True
    keep_litematic: bool = True
    tempo_bpm: float = 0.0        # 0 = 由音频时长估算
    song_name: str = ""
    author: str = "mp3-nbs-converter"


@dataclass
class ConvertResult:
    """转换产物。"""

    nbs_path: str | None = None
    midi_path: str | None = None
    litematic_path: str | None = None
    instrumental_path: str | None = None
    stats: dict = field(default_factory=dict)


def _safe_name(name: str) -> str:
    """把歌曲名转成文件安全的 ASCII 名。"""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch == " ":
            keep.append("_")
    out = "".join(keep)[:60].strip("_")
    return out or "song"


def _cache_key(input_path: str, samples: np.ndarray, model: str, max_seconds: float) -> str:
    """分离缓存键：音频内容 + 模型 + 截断长度。

    不依赖文件路径，所以同一首歌改名/移动后仍能命中。
    """
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(str(samples.shape).encode())
    h.update(np.ascontiguousarray(samples).tobytes())
    h.update(model.encode())
    h.update(f"{max_seconds:.3f}".encode())
    return h.hexdigest()


def _load_cached_accomp(key: str) -> np.ndarray | None:
    """读取缓存的伴奏数组，失败返回 None。"""
    try:
        from .paths import cache_dir

        p = cache_dir() / f"{key}_accomp.npy"
        if p.is_file():
            return np.load(p)
    except Exception:  # noqa: BLE001
        pass
    return None


def _save_cached_accomp(key: str, accomp: np.ndarray) -> None:
    """写入伴奏缓存（超过 200 MB 不缓存，避免撑爆磁盘）。"""
    try:
        from .paths import cache_dir

        if accomp.nbytes > 200 * 1024 * 1024:
            return
        d = cache_dir()
        np.save(d / f"{key}_accomp.npy", accomp.astype(np.float32))
        # 顺手清理：只留最近 12 个缓存
        files = sorted(d.glob("*_accomp.npy"), key=lambda f: f.stat().st_mtime, reverse=True)
        for old in files[12:]:
            old.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _estimate_bpm(duration: float, notes: list) -> float:
    """由音符间隔估算 BPM（取最常见的间隔倒数）。"""
    if len(notes) < 4:
        return 120.0
    intervals = []
    prev_end = None
    for n in notes[:400]:
        if prev_end is not None:
            d = n.start - prev_end
            if 0.05 < d < 2.0:
                intervals.append(d)
        prev_end = n.start + n.duration
    if not intervals:
        return 120.0
    from collections import Counter

    # 量化到 16 分音符网格
    grid = [round(i * 0.0625, 4) for i in range(1, 33)]
    best = min(grid, key=lambda g: min(abs(i - g) for i in intervals))
    bpm = 60.0 / best / 4.0 * 4.0  # 一个四分音符 = best
    return max(60.0, min(200.0, bpm))


def _split_tracks(notes, bass_split_midi: int):
    """按 MIDI 音高把音符分成低音轨与主旋律轨。"""
    bass, melody = [], []
    for n in notes:
        (bass if n.midi < bass_split_midi else melody).append(n)
    return bass, melody


def _to_layout_notes(notes, instrument: str):
    """音符列表 → {game_tick: [(instrument, nbs_key)]}。"""
    out: dict[int, list[tuple[str, int]]] = {}
    for n in notes:
        nbs_key = cl.fold_to_nbs_range(int(n.midi))
        if cl.key_to_mc_note(nbs_key) is None:
            continue
        tick = int(round(n.start * GAME_TICKS_PER_SEC))
        if tick < 0:
            continue
        out.setdefault(tick, []).append((instrument, nbs_key))
    return out


def convert_audio(
    input_path: str,
    out_dir: str,
    options: ConvertOptions | None = None,
    progress=None,
) -> ConvertResult:
    """执行完整转换流程。

    progress: 可调用对象 progress(stage: str, ratio: float)
    """
    opts = options or ConvertOptions()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def report(stage: str, ratio: float) -> None:
        if progress is not None:
            try:
                progress(stage, max(0.0, min(1.0, float(ratio))))
            except Exception:  # noqa: BLE001 —— 进度回调不应影响转换
                pass

    t_start = time.time()
    report("解码音频", 0.02)

    audio: Audio = decode(input_path, sample_rate=44100, mono=False)
    samples = audio.samples
    if samples.ndim == 1:
        samples = samples[None, :]
    sr = audio.sample_rate

    if opts.max_seconds and opts.max_seconds > 0:
        n = int(opts.max_seconds * sr)
        if n < samples.shape[1]:
            samples = samples[:, :n]
    duration = samples.shape[1] / sr
    if duration < 0.05:
        raise ValueError("音频太短或无法解码")

    # --- 去人声 ---
    instrumental = samples
    vocals = None
    vocal_fallback = False
    if opts.remove_vocals:
        report("分离人声", 0.08)
        ckey = _cache_key(input_path, samples, opts.vocal_model, opts.max_seconds)
        cached = _load_cached_accomp(ckey)
        if cached is not None and cached.shape == samples.shape:
            instrumental = cached
            report("分离人声", 0.30)
            print("[去人声] 命中缓存，跳过分离（调参数时省时间）")
        else:
            from .vocal_separate import LAST_WAS_FALLBACK, separate_vocals

            def _sep_progress(ratio: float) -> None:
                # 分离阶段进度映射到 0.08 ~ 0.30
                report("分离人声", 0.08 + 0.22 * max(0.0, min(1.0, float(ratio))))

            accomp, voc = separate_vocals(samples, sr, opts.vocal_model, progress=_sep_progress)
            instrumental = accomp
            vocals = voc
            report("分离人声", 0.30)
            if LAST_WAS_FALLBACK:
                # 降级结果不入缓存，避免污染（模型修好后还能重算）
                vocal_fallback = True
                print("[去人声] 本次为降级结果，不写入缓存")
            else:
                _save_cached_accomp(ckey, accomp)

    # 整体静音判定：原曲就是静音时，分离模型会输出浮点噪声，
    # 直接判定为静音，避免把数值噪声当成音符。
    src_peak = float(np.abs(samples).max()) if samples.size else 0.0
    if src_peak < 1e-4:
        print("[转录] 输入为静音，跳过音高识别")
        from .transcribe import Transcription

        transcription = Transcription(source="silent")
        notes = []
        melody_notes = []
        bass_notes = []
        bpm = 120.0
    else:
        # --- 音高识别 ---
        report("识别音高", 0.34)
        from .transcribe import transcribe

        transcription = transcribe(
            instrumental,
            sr,
            prefer_model=True,
            onset_threshold=opts.onset_threshold,
            frame_threshold=opts.frame_threshold,
            minimum_note_length_ms=opts.min_note_ms,
            melodia_trick=opts.melodia_trick,
        )
        notes = transcription.notes
        report("识别音高", 0.68)

        if not notes:
            raise ValueError("没有识别到任何音符，请尝试调整起音阈值或更换音频")

    # --- 写伴奏 wav ---
    instrumental_path = None
    try:
        ip = str(out_path / f"{_safe_name(opts.song_name or Path(input_path).stem)}_instrumental.wav")
        peak = float(np.abs(instrumental).max())
        if peak > 0:
            instrumental = instrumental / peak * 0.95
        write_wav(ip, instrumental, sr)
        instrumental_path = ip
    except Exception:  # noqa: BLE001
        instrumental_path = None

    song_name = opts.song_name or Path(input_path).stem
    base = _safe_name(song_name)
    bpm = opts.tempo_bpm if opts.tempo_bpm > 0 else _estimate_bpm(duration, notes)

    bass_notes, melody_notes = _split_tracks(notes, opts.bass_split_midi)
    main_inst = nbs_writer.normalize_instrument(opts.instrument)
    bass_inst = nbs_writer.normalize_instrument(opts.bass_instrument)

    # --- NBS ---
    nbs_path = None
    if opts.keep_nbs:
        report("生成音符盒", 0.74)
        tracks: list[nbs_writer.Track] = []
        # NBS tick = 游戏刻 / 2（10 tick/秒 vs 20 刻/秒）
        if melody_notes:
            tracks.append(
                nbs_writer.Track(
                    "主旋律",
                    [
                        nbs_writer.Note(
                            tick=int(round(n.start * NBS_TICKS_PER_SEC)),
                            layer=0,
                            key=cl.fold_to_nbs_range(int(n.midi)),
                            velocity=max(1, min(127, int(round((n.velocity or 0.7) * 100)))),
                            instrument=main_inst,
                        )
                        for n in melody_notes
                    ],
                )
            )
        if bass_notes:
            tracks.append(
                nbs_writer.Track(
                    "低音",
                    [
                        nbs_writer.Note(
                            tick=int(round(n.start * NBS_TICKS_PER_SEC)),
                            layer=len(tracks),
                            key=cl.fold_to_nbs_range(int(n.midi)),
                            velocity=max(1, min(127, int(round((n.velocity or 0.7) * 100)))),
                            instrument=bass_inst,
                        )
                        for n in bass_notes
                    ],
                )
            )
        if not tracks:
            tracks = [nbs_writer.Track("主旋律", [])]
        nbs_path = str(out_path / f"{base}.nbs")
        nbs_writer.write_nbs_file(
            nbs_path,
            tracks,
            song_name=song_name,
            author=opts.author,
            description="Generated by MP3 to NBS Converter",
            tempo=NBS_TICKS_PER_SEC,
        )
    report("生成音符盒", 0.84)

    # --- MIDI ---
    midi_path = None
    if opts.keep_midi:
        report("写出 MIDI", 0.87)
        midi_tracks: list[midi_writer.MidiTrack] = []
        if melody_notes:
            prog, tname = midi_writer.MIDI_PROGRAMS.get(main_inst, (0, main_inst))
            midi_tracks.append(
                midi_writer.MidiTrack(
                    tname,
                    [
                        midi_writer.MidiNote(
                            n.start, n.duration, int(n.midi),
                            max(1, min(127, int(round((n.velocity or 0.7) * 100)))), 0,
                        )
                        for n in melody_notes
                    ],
                    prog, 0,
                )
            )
        if bass_notes:
            prog, tname = midi_writer.MIDI_PROGRAMS.get(bass_inst, (42, bass_inst))
            midi_tracks.append(
                midi_writer.MidiTrack(
                    tname,
                    [
                        midi_writer.MidiNote(
                            n.start, n.duration, int(n.midi),
                            max(1, min(127, int(round((n.velocity or 0.7) * 100)))), 1,
                        )
                        for n in bass_notes
                    ],
                    prog, 1,
                )
            )
        midi_path = str(out_path / f"{base}.midi")
        midi_writer.write_midi_file(
            midi_path, midi_tracks, song_name=song_name, bpm=bpm
        )
    report("写出 MIDI", 0.92)

    # --- litematic ---
    litematic_path = None
    if opts.keep_litematic:
        report("生成蓝图", 0.94)
        mel_layout = _to_layout_notes(melody_notes, main_inst)
        bass_layout = _to_layout_notes(bass_notes, bass_inst)
        blocks: dict[tuple[int, int, int], dict] = {}
        if mel_layout:
            blocks.update(
                cl.build_compact_blocks(
                    mel_layout, coarse=opts.coarse, wrap=opts.wrap, gap=opts.gap
                )
            )
        if bass_layout:
            bass_blocks = cl.build_compact_blocks(
                bass_layout, coarse=opts.coarse, wrap=opts.wrap, gap=opts.gap
            )
            # 低音轨沿 z 轴偏移，避免与主旋律重叠
            off_z = (max((p[2] for p in blocks), default=0) + 4) if blocks else 0
            for (x, y, z), st in bass_blocks.items():
                blocks[(x, y, z + off_z)] = st
        litematic_path = str(out_path / f"{base}.litematic")
        with open(litematic_path, "wb") as fh:
            fh.write(
                cl.to_litematic(
                    blocks,
                    name=song_name,
                    description="由 MP3 → NBS 转换器生成",
                    author=opts.author,
                )
            )
    report("完成", 1.0)

    stats = {
        "duration": round(duration, 2),
        "notes": len(notes),
        "melody_notes": len(melody_notes),
        "bass_notes": len(bass_notes),
        "removed_vocals": bool(opts.remove_vocals),
        "vocal_model": opts.vocal_model if opts.remove_vocals else None,
        "vocal_fallback": bool(vocal_fallback),
        "layers": 2 if (melody_notes and bass_notes) else (1 if notes else 0),
        "bpm": round(bpm, 1),
        "instrument": main_inst,
        "bass_instrument": bass_inst,
        "blocks": len(blocks) if opts.keep_litematic else 0,
        "elapsed": round(time.time() - t_start, 2),
        "sample_rate": sr,
    }
    if vocals is not None:
        stats["instrumental_seconds"] = round(duration, 2)

    return ConvertResult(
        nbs_path=nbs_path,
        midi_path=midi_path,
        litematic_path=litematic_path,
        instrumental_path=instrumental_path,
        stats=stats,
    )
