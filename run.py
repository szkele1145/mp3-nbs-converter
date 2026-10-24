"""MP3 → 音符盒转换器 · 命令行入口 / Web 服务入口。

用法：
    python run.py                                启动 Web 服务（默认 http://127.0.0.1:8760）
    python run.py --port 9000                    换端口
    python run.py --cli 输入.mp3 -o 输出目录      命令行转换
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# 资源友好：必须在 numpy / onnxruntime 导入前设置线程上限
try:
    from core.config import apply_env_defaults

    apply_env_defaults()
except Exception:  # noqa: BLE001 —— 打包/路径异常时也要能跑
    pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="MP3 → Minecraft 音符盒(.nbs) / MIDI / Litematica 转换器，支持智能去人声",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python run.py
  python run.py --port 9000 --host 127.0.0.1
  python run.py --cli song.mp3 -o ./out --no-vocals --instrument PIANO
""",
    )
    p.add_argument("--cli", metavar="音频文件", help="以命令行模式转换指定文件")
    p.add_argument("-o", "--output", metavar="目录", default="output", help="输出目录（默认 output）")
    p.add_argument("--host", default="127.0.0.1", help="Web 服务监听地址（默认 127.0.0.1）")
    p.add_argument("--port", type=int, default=8760, help="Web 服务端口（默认 8760）")
    p.add_argument("--no-vocals", action="store_true", help="不去人声，直接用原始音频")
    p.add_argument("--model", default="htdemucs_ft_vocals",
                   help="人声分离模型：htdemucs_ft_vocals（质量最好）/ htdemucs / UVR_MDXNET_Main（最省内存 1.5GB）")
    p.add_argument("--onset", type=float, default=0.5, help="起音阈值（默认 0.5）")
    p.add_argument("--frame", type=float, default=0.3, help="帧阈值（默认 0.3）")
    p.add_argument("--min-note-ms", type=float, default=127.7, help="音符最短长度毫秒（默认 127.7）")
    p.add_argument("--instrument", default="HARP", help="主旋律乐器（默认 HARP）")
    p.add_argument("--bass-instrument", default="DOUBLE_BASS", help="低音轨乐器（默认 DOUBLE_BASS）")
    p.add_argument("--coarse", type=int, default=0, help="中继器粒度 0(不限)/1-4")
    p.add_argument("--wrap", type=int, default=0, help="每行最大列数，0 = 不换行")
    p.add_argument("--gap", type=int, default=0, help="轨道间距")
    p.add_argument("--max-seconds", type=float, default=0.0, help="只转换前 N 秒，0 = 不限")
    p.add_argument("--no-nbs", action="store_true", help="不输出 .nbs")
    p.add_argument("--no-midi", action="store_true", help="不输出 .midi")
    p.add_argument("--no-litematic", action="store_true", help="不输出 .litematic")
    p.add_argument("--threads", type=int, default=0,
                   help="推理线程数（默认自动，约为核心数/3，留资源给游戏）")
    p.add_argument("--high-memory", action="store_true",
                   help="关闭低内存模式（更快但更吃内存）")
    p.add_argument("--normal-priority", action="store_true",
                   help="转换时不降低进程优先级（更快但会抢游戏资源）")
    return p


def _run_cli(args: argparse.Namespace) -> int:
    from core.config import (
        apply_env_defaults,
        lower_process_priority,
        restore_process_priority,
        set_low_memory,
        set_threads,
        threads,
    )
    from core.pipeline import ConvertOptions, convert_audio

    if args.threads and args.threads > 0:
        set_threads(args.threads)
    if args.high_memory:
        set_low_memory(False)
    apply_env_defaults()

    src = Path(args.cli)
    if not src.is_file():
        print(f"[错误] 文件不存在：{src}", file=sys.stderr)
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    options = ConvertOptions(
        remove_vocals=not args.no_vocals,
        vocal_model=args.model,
        onset_threshold=args.onset,
        frame_threshold=args.frame,
        min_note_ms=args.min_note_ms,
        instrument=args.instrument,
        bass_instrument=args.bass_instrument,
        coarse=args.coarse,
        wrap=args.wrap,
        gap=args.gap,
        max_seconds=args.max_seconds,
        keep_nbs=not args.no_nbs,
        keep_midi=not args.no_midi,
        keep_litematic=not args.no_litematic,
        song_name=src.stem,
    )

    last_stage = ""

    def progress(stage: str, ratio: float) -> None:
        nonlocal last_stage
        if stage != last_stage:
            last_stage = stage
            print(f"  → {stage} …", flush=True)

    print(f"开始转换：{src}")
    print(f"输出目录：{out_dir.resolve()}")
    print(f"资源设置：线程 {threads()} · "
          f"{'低内存模式' if not args.high_memory else '高内存模式'}")
    if not args.normal_priority:
        lower_process_priority()
    try:
        result = convert_audio(str(src), str(out_dir), options, progress)
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] 转换失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if not args.normal_priority:
            restore_process_priority()

    print("\n转换完成：")
    for label, path in (
        ("音符盒 (.nbs)", result.nbs_path),
        ("MIDI", result.midi_path),
        ("蓝图 (.litematic)", result.litematic_path),
        ("伴奏音频 (.wav)", result.instrumental_path),
    ):
        if path:
            size = Path(path).stat().st_size
            print(f"  {label:<20} {path}  ({size:,} 字节)")

    stats = result.stats
    print("\n统计：")
    print(f"  时长 {stats.get('duration')} 秒 · 音符 {stats.get('notes')} 个"
          f"（主旋律 {stats.get('melody_notes')} / 低音 {stats.get('bass_notes')}）")
    print(f"  去人声：{'是' if stats.get('removed_vocals') else '否'}"
          f"{'（' + str(stats.get('vocal_model')) + '）' if stats.get('removed_vocals') else ''}")
    print(f"  轨道 {stats.get('layers')} 条 · BPM {stats.get('bpm')} · 用时 {stats.get('elapsed')} 秒")
    return 0


def _run_web(args: argparse.Namespace) -> int:
    import uvicorn

    from web.server import app

    banner = f"""
  ╭──────────────────────────────────────────────╮
  │   MP3 → 音符盒转换器 · Web 服务已启动          │
  ╰──────────────────────────────────────────────╯
    地址：http://{args.host}:{args.port}
    按 Ctrl+C 停止
"""
    print(banner)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main() -> int:
    args = _build_parser().parse_args()
    if args.cli:
        return _run_cli(args)
    return _run_web(args)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
