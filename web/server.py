# -*- coding: utf-8 -*-
"""MP3 → 音符盒转换器 · FastAPI 后端。

提供：
    GET  /                     前端页面（web/index.html）
    GET  /static/*             静态资源
    GET  /api/health           健康检查
    GET  /api/models           可用分离模型 + 乐器表
    POST /api/convert          上传音频并启动转换任务（后台线程执行）
    GET  /api/progress         当前任务状态
    GET  /api/download/{kind}  下载 nbs / midi / litematic / instrumental

约定：
    * 转换核心在 core.pipeline（同事并行实现），此处只按接口契约调用。
    * core.vocal_separate 可能暂时不存在，导入失败时退回内置模型列表。
    * 不在 import 阶段加载 ONNX 模型（太慢），模型加载交给转换线程。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from core.paths import output_dir as _out_dir  # noqa: E402
from core.paths import resource_dir as _res_dir  # noqa: E402
from core.paths import upload_dir as _up_dir  # noqa: E402

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

VERSION = "1.1.0"
WEB_DIR = _res_dir()
WORK_DIR = _out_dir().parent
UPLOAD_DIR = _up_dir()
OUTPUT_DIR = _out_dir()

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
CHUNK_SIZE = 1024 * 1024

#: 下载种类 → (媒体类型, 默认扩展名)
DOWNLOAD_KINDS: Dict[str, tuple] = {
    "nbs": ("application/octet-stream", ".nbs"),
    "midi": ("audio/midi", ".mid"),
    "litematic": ("application/octet-stream", ".litematic"),
    "instrumental": ("audio/wav", ".wav"),
}

#: 乐器 id → 中文名（不 import core.compact_layout，避免耦合）
INSTRUMENT_NAMES: Dict[str, str] = {
    "HARP": "竖琴",
    "PIANO": "钢琴",
    "FLUTE": "长笛",
    "BELL": "钟",
    "DOUBLE_BASS": "大提琴",
    "GUITAR": "木吉他",
    "BASS_DRUM": "低音鼓",
    "SNARE_DRUM": "军鼓",
    "CLICK": "踩镲",
    "BANJO": "班卓琴",
    "TRUMPET": "铜管",
    "COW_BELL": "牛铃",
    "XYLOPHONE": "木琴",
    "IRON_XYLOPHONE": "铁木琴",
    "BIT": "滴答",
    "PLING": "拨弦",
    "DIDGERIDOO": "竹笛",
    "CHIME": "钟琴",
}

#: core.vocal_separate 不可用时的兜底模型列表
DEFAULT_MODELS: List[Dict[str, str]] = [
    {"id": "htdemucs", "name": "HTDemucs 四源分离（推荐）"},
    {"id": "UVR_MDXNET_Main", "name": "UVR MDX-NET"},
]

#: 前端 / CLI 可能出现的模型 id 别名 → 规范 id
MODEL_ALIASES: Dict[str, str] = {
    "htdemucs": "htdemucs",
    "htdemucs_ft_vocals": "htdemucs_ft_vocals",
    "htdemucs_ft": "htdemucs_ft_vocals",
    "htdemucs_embedded": "htdemucs_embedded",
    "uvr_mdxnet_main": "UVR_MDXNET_Main",
    "mdxnet": "UVR_MDXNET_Main",
}

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "t", "是", "开", "开启", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "f", "否", "关", "关闭", "disable", "disabled"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _to_bool(value: Any, default: bool = False) -> bool:
    """把表单里的字符串稳健地转成 bool。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _normalize_model(model_id: str) -> str:
    """模型 id 归一化（大小写不敏感），未知值原样保留。"""
    text = (model_id or "").strip()
    return MODEL_ALIASES.get(text.lower(), text)


def _safe_filename(name: str, fallback: str = "song") -> str:
    """生成安全的下载文件名。

    保留中文（现代浏览器和 Windows 都支持），只去掉路径分隔符、
    控制字符和 Windows 保留字符。空名时用 fallback。
    """
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    # Windows 非法字符：< > : " / \ | ? *  以及控制字符
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem)
    # 去掉首尾空白和点（Windows 不允许以点结尾的文件名）
    stem = stem.strip(" ._")
    # Windows 保留设备名
    if stem.upper() in {
        "CON", "PRN", "AUX", "NUL",
        *{f"COM{i}" for i in range(1, 10)},
        *{f"LPT{i}" for i in range(1, 10)},
    }:
        stem = "_" + stem
    if not stem:
        stem = fallback
    return stem[:80]


def _find_binary(name: str) -> Optional[str]:
    """在 PATH 与 tools/ 下寻找可执行文件（Windows 友好）。"""
    found = shutil.which(name)
    if found:
        return found
    for suffix in (".exe", ""):
        candidate = PROJ_ROOT / "tools" / (name + suffix)
        if candidate.is_file():
            return str(candidate)
    return None


def _probe_duration(path: str) -> Optional[float]:
    """探测音频时长（秒）。ffprobe 优先，其次 librosa；都失败则返回 None。"""
    exe = _find_binary("ffprobe")
    if exe:
        try:
            proc = subprocess.run(
                [
                    exe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    path,
                ],
                capture_output=True,
                timeout=60,
            )
            if proc.returncode == 0:
                text = proc.stdout.decode("utf-8", "replace").strip()
                if text:
                    value = float(text)
                    if value > 0:
                        return value
        except Exception:  # noqa: BLE001 - 探测失败不致命
            pass

    try:
        import librosa  # 延迟导入：librosa 启动较慢

        value = float(librosa.get_duration(path=path))
        if value > 0:
            return value
    except Exception:  # noqa: BLE001 - 探测失败不致命
        pass

    return None


def _load_models() -> List[Dict[str, str]]:
    """取可用的分离模型列表；core.vocal_separate 缺失时返回内置列表。"""
    try:
        from core.vocal_separate import list_models  # type: ignore

        models = list_models()
        if isinstance(models, list) and models:
            cleaned: List[Dict[str, str]] = []
            for item in models:
                if isinstance(item, dict) and item.get("id"):
                    cleaned.append(
                        {
                            "id": str(item["id"]),
                            "name": str(item.get("name") or item["id"]),
                        }
                    )
            if cleaned:
                return cleaned
    except Exception:  # noqa: BLE001 - 模块可能尚不存在
        pass
    return [dict(m) for m in DEFAULT_MODELS]


# ---------------------------------------------------------------------------
# 全局任务状态（线程安全）
# ---------------------------------------------------------------------------


#: 各阶段在全局进度中的区间（与 core.pipeline 的 report 比例一致）
#: 用于按「本阶段完成度」估算 ETA，避免用全局进度导致高估。
STAGE_RANGES: Dict[str, tuple] = {
    "解码音频": (0.00, 0.08),
    "分离人声": (0.08, 0.30),
    "识别音高": (0.30, 0.68),
    "生成音符盒": (0.68, 0.86),
    "写出 MIDI": (0.86, 0.92),
    "生成蓝图": (0.92, 1.00),
    "完成": (1.00, 1.00),
}


@dataclass
class JobState:
    """保存当前唯一任务的运行状态。"""

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    status: str = "idle"           # idle | running | done | error
    progress: float = 0.0          # 0..1
    stage: str = ""                # 中文阶段名
    message: str = ""              # 补充说明
    error: Optional[str] = None    # 中文错误信息
    job_id: str = ""               # 当前任务 ID
    result: Optional[Dict[str, Any]] = None   # 展示用结果
    files: Dict[str, str] = field(default_factory=dict)   # kind → 绝对路径
    names: Dict[str, str] = field(default_factory=dict)   # kind → ASCII 文件名
    eta_seconds: Optional[float] = None
    _start_time: float = 0.0
    _last_update_time: float = 0.0
    _stage_start_time: float = 0.0
    _stage_start_ratio: float = 0.0

    def begin(self, job_id: str) -> bool:
        """尝试占坑；已有任务在跑时返回 False。"""
        with self._lock:
            if self.status == "running":
                return False
            now = time.time()
            self.status = "running"
            self.progress = 0.0
            self.stage = "准备中"
            self.message = "正在准备转换任务…"
            self.error = None
            self.job_id = job_id
            self.result = None
            self.files = {}
            self.names = {}
            self.eta_seconds = None
            self._start_time = now
            self._last_update_time = now
            self._stage_start_time = now
            self._stage_start_ratio = 0.0
            return True

    def update(self, stage: str = "", ratio: float = 0.0, job_id: str = "") -> None:
        with self._lock:
            if job_id and self.job_id and self.job_id != job_id:
                return  # 旧任务的回调，忽略
            now = time.time()
            if stage and stage != self.stage:
                # 换了阶段：重置阶段计时，ETA 改用「本阶段进度」估算
                self.stage = str(stage)
                self._stage_start_time = now
                self._stage_start_ratio = self.progress
            value = _to_float(ratio, self.progress)
            self.progress = min(1.0, max(0.0, value))
            self._last_update_time = now
            eta = self._estimate_eta(now)
            self.eta_seconds = eta
            if eta is None:
                self.message = f"{self.stage} · {int(self.progress * 100)}%"
            else:
                if eta < 60:
                    self.message = f"{self.stage} · {int(self.progress * 100)}% · 预计剩余 {int(eta)}s"
                else:
                    m = int(eta // 60)
                    s = int(eta % 60)
                    self.message = f"{self.stage} · {int(self.progress * 100)}% · 预计剩余 {m}m{s:02d}s"

    def _estimate_eta(self, now: float) -> Optional[float]:
        """估算剩余时间。

        用「本阶段完成度」估算：分离阶段占全程 90% 时间，
        若用全局进度算会严重高估剩余时间。
        阶段区间由 STAGE_RANGES 给出。
        """
        if self.status != "running" or self._start_time <= 0:
            return None
        if self.progress >= 0.995:
            return None

        span = STAGE_RANGES.get(self.stage)
        stage_elapsed = now - self._stage_start_time if self._stage_start_time > 0 else 0.0

        if span:
            lo, hi = span
            width = max(hi - lo, 1e-6)
            done = min(1.0, max(0.0, (self.progress - lo) / width))
            if done > 0.01 and stage_elapsed > 0.5:
                # 本阶段剩余时间
                rem = stage_elapsed * (1.0 - done) / done
                # 之后还有多少全局进度没走，按「分离阶段速度」折算
                tail_ratio = max(0.0, 1.0 - hi)
                # 分离阶段大约占总时间 85%，其余阶段按比例折算
                tail = tail_ratio * stage_elapsed / 0.22 * 0.18
                total = rem + tail
                if total >= 1.0:
                    return max(1.0, total)
                return None

        # 兜底：用全局进度
        elapsed = now - self._start_time
        if elapsed > 0.5 and self.progress > 0.02:
            total_est = elapsed / max(self.progress, 1e-6)
            rem = total_est - elapsed
            if rem >= 1.0:
                return max(1.0, rem)
        return None

    def fail_engine(self, job_id: str, reason: str) -> None:
        """排布引擎不可用：标记成待用户确认状态，而不是普通失败。"""
        with self._lock:
            if self.job_id and job_id and self.job_id != job_id:
                return
            self.status = "need_engine"
            self.stage = "需要排布引擎"
            self.message = reason or "本地排布引擎不可用"
            self.error = None
            self.eta_seconds = None

    def finish(self, job_id: str, result: Dict[str, Any],
               files: Dict[str, str], names: Dict[str, str]) -> None:
        with self._lock:
            if self.job_id != job_id:
                return
            self.status = "done"
            self.progress = 1.0
            self.stage = "转换完成"
            self.message = "全部文件已生成"
            self.error = None
            self.result = result
            self.files = dict(files)
            self.names = dict(names)
            self.eta_seconds = None

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            if self.job_id and job_id and self.job_id != job_id:
                return
            self.status = "error"
            self.stage = "转换失败"
            self.message = error
            self.error = error
            self.eta_seconds = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # 实时重算 ETA：分离等长耗时阶段进度不更新，但时间在走
            eta = self._estimate_eta(time.time())
            if eta is None:
                eta = self.eta_seconds
            return {
                "status": self.status,
                "progress": round(self.progress, 4),
                "stage": self.stage,
                "message": self.message,
                "error": self.error,
                "job_id": self.job_id,
                "result": self.result,
                "eta": eta,
            }


JOB = JobState()


# ---------------------------------------------------------------------------
# 表单 → ConvertOptions
# ---------------------------------------------------------------------------

#: 表单字段 → ConvertOptions 字段的别名映射（按顺序取第一个存在的）
FORM_ALIASES: Dict[str, List[str]] = {
    "remove_vocals": ["remove_vocals", "separate_vocals", "no_vocals", "vocal_separation"],
    "vocal_model": ["vocal_model", "model", "vocal_model_id", "separate_model"],
    "onset_threshold": ["onset_threshold", "onset"],
    "frame_threshold": ["frame_threshold", "frame"],
    "min_note_ms": ["min_note_ms", "min_note_length_ms", "min_note_ms", "minlen"],
    "melodia_trick": ["melodia_trick", "melodia"],
    "instrument": ["instrument"],
    "bass_instrument": ["bass_instrument", "bass"],
    "coarse": ["coarse"],
    "wrap": ["wrap"],
    "gap": ["gap"],
    "max_seconds": ["max_seconds", "max_sec", "max_seconds_limit"],
    "threads": ["threads", "infer_threads", "n_threads"],
    "bass_split_midi": ["bass_split_midi", "bass_split"],
    "keep_nbs": ["keep_nbs", "output_nbs", "nbs"],
    "keep_midi": ["keep_midi", "output_midi", "midi"],
    "keep_litematic": ["keep_litematic", "output_litematic", "litematic"],
}


def _pick(form: Dict[str, Any], key: str) -> Any:
    """按别名列表从表单里取值。"""
    for alias in FORM_ALIASES.get(key, [key]):
        if alias in form:
            value = form[alias]
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def _build_options(form: Dict[str, Any]):
    """把表单字典转成 core.pipeline.ConvertOptions 实例。"""
    from core.pipeline import ConvertOptions  # 延迟导入：core 由同事实现

    import dataclasses

    desired: Dict[str, Any] = {
        "remove_vocals": _to_bool(_pick(form, "remove_vocals"), True),
        "vocal_model": _normalize_model(str(_pick(form, "vocal_model") or "htdemucs_ft_vocals")),
        "onset_threshold": _to_float(_pick(form, "onset_threshold"), 0.5),
        "frame_threshold": _to_float(_pick(form, "frame_threshold"), 0.3),
        "min_note_ms": _to_float(_pick(form, "min_note_ms"), 127.7),
        "melodia_trick": _to_bool(_pick(form, "melodia_trick"), True),
        "instrument": str(_pick(form, "instrument") or "AUTO"),
        "bass_instrument": str(_pick(form, "bass_instrument") or "AUTO"),
        "auto_preset": _to_bool(_pick(form, "auto_preset"), False),
        "allow_builtin_layout": _to_bool(_pick(form, "allow_builtin_layout"), False),
        "coarse": _to_int(_pick(form, "coarse"), 0),
        "wrap": _to_int(_pick(form, "wrap"), 0),
        "gap": _to_int(_pick(form, "gap"), 0),
        "max_seconds": _to_float(_pick(form, "max_seconds"), 0.0),
        "bass_split_midi": _to_int(_pick(form, "bass_split_midi"), 55),
        "keep_nbs": _to_bool(_pick(form, "keep_nbs"), True),
        "keep_midi": _to_bool(_pick(form, "keep_midi"), True),
        "keep_litematic": _to_bool(_pick(form, "keep_litematic"), True),
    }

    # 只传 ConvertOptions 真正声明了的字段，避免签名不一致直接 TypeError
    try:
        valid = {f.name for f in dataclasses.fields(ConvertOptions)}
    except TypeError:
        valid = set(desired)
    kwargs = {k: v for k, v in desired.items() if k in valid}
    return ConvertOptions(**kwargs)


# ---------------------------------------------------------------------------
# 转换任务执行体（跑在后台线程里）
# ---------------------------------------------------------------------------


def _run_conversion(job_id: str, input_path: str, out_dir: str, form: Dict[str, Any]) -> None:
    """在工作线程中执行转换；异常统一写回 JOB 状态。"""
    lowered = False
    prev_threads = None
    try:
        from core.config import lower_process_priority, restore_process_priority, set_threads
        from core.pipeline import convert_audio  # 延迟导入，避免启动即加载重依赖

        # 推理线程数（0 = 保持全局默认）
        threads = _to_int(_pick(form, "threads"), 0)
        if threads > 0:
            from core.config import threads as current_threads

            prev_threads = current_threads()
            set_threads(threads)

        options = _build_options(form)

        def _progress(stage: str = "", ratio: float = 0.0, *args: Any, **kwargs: Any) -> None:
            JOB.update(stage=stage, ratio=ratio, job_id=job_id)

        JOB.update(stage="解码音频", ratio=0.02, job_id=job_id)

        # 先探一下时长（失败无所谓，交给 pipeline 处理）
        duration = _probe_duration(input_path)
        if duration:
            JOB.update(stage="解码音频", ratio=0.05, job_id=job_id)

        # 转换期间降低进程优先级：用户在玩游戏时不会被顶掉
        if not _to_bool(_pick(form, "normal_priority"), False):
            lower_process_priority()
            lowered = True

        result_obj = convert_audio(input_path, out_dir, options, progress=_progress)

        result, files, names = _collect_outputs(result_obj, job_id)
        JOB.finish(job_id, result, files, names)
    except Exception as exc:  # noqa: BLE001 - 需要把任何异常转成友好提示
        # 排布引擎缺失：不当作普通错误，标出类型让界面弹窗询问
        from core.pipeline import LayoutEngineMissing

        if isinstance(exc, LayoutEngineMissing):
            print(f"[排布] 引擎不可用，等待用户选择：{exc.reason}")
            JOB.fail_engine(job_id, exc.reason)
            return
        traceback.print_exc()
        JOB.fail(job_id, _friendly_error(exc))
    finally:
        if lowered:
            try:
                from core.config import restore_process_priority

                restore_process_priority()
            except Exception:  # noqa: BLE001
                pass
        if prev_threads is not None:
            # 恢复线程设置，避免本次任务的设置影响下一个任务
            try:
                from core.config import set_threads

                set_threads(prev_threads)
            except Exception:  # noqa: BLE001
                pass


def _friendly_error(exc: Exception) -> str:
    """把常见异常转成用户看得懂的中文提示。"""
    msg = str(exc)
    if isinstance(exc, ValueError) and msg:
        # pipeline 里的 ValueError 都是已经写好的中文提示
        if any(k in msg for k in ("音频", "音符", "阈值", "太长", "太短")):
            return msg
    low = msg.lower()
    if "ffmpeg" in low or "codec" in low or "decoder" in low:
        return "无法解码该音频文件，请换用 MP3 / WAV 格式后重试。"
    if "permission" in low or "access is denied" in low:
        return "没有写入权限，请换一个输出目录后重试。"
    if "no space left" in low or "disk" in low:
        return "磁盘空间不足，请清理后重试。"
    if "memory" in low or "out of memory" in low:
        return "内存不足，请关闭其他程序或切换到「游戏模式」后重试。"
    if "model" in low and ("not found" in low or "no such" in low):
        return "分离模型文件缺失，请把 .onnx 模型放到 models/ 目录后重试。"
    return f"转换失败：{msg}" if msg else "转换失败，请查看后端日志。"


def _collect_outputs(result_obj: Any, job_id: str):
    """从 ConvertResult 里挑出可下载的文件。"""
    files: Dict[str, str] = {}
    names: Dict[str, str] = {}
    display: Dict[str, Any] = {}
    sizes: Dict[str, int] = {}

    for kind, attr, default_ext in (
        ("nbs", "nbs_path", ".nbs"),
        ("midi", "midi_path", ".mid"),
        ("litematic", "litematic_path", ".litematic"),
        ("instrumental", "instrumental_path", ".wav"),
    ):
        raw = getattr(result_obj, attr, None)
        if not raw:
            continue
        path = os.path.abspath(str(raw))
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1] or default_ext
        ascii_name = _safe_filename(os.path.splitext(os.path.basename(path))[0]) + ext
        files[kind] = path
        names[kind] = ascii_name
        display[kind] = ascii_name
        try:
            sizes[kind] = os.path.getsize(path)
        except OSError:
            pass

    if sizes:
        display["sizes"] = sizes

    stats = getattr(result_obj, "stats", None)
    stats = dict(stats) if isinstance(stats, dict) else {}
    display["stats"] = stats
    return display, files, names


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MP3 → 音符盒转换器",
    description="把歌曲转换为 Minecraft 音符盒 / MIDI / Litematica 蓝图，支持智能去人声。",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回前端页面。"""
    page = WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="前端页面缺失：web/index.html")
    return FileResponse(str(page), media_type="text/html; charset=utf-8")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """健康检查。"""
    return {"ok": True, "version": VERSION}


@app.get("/api/engine")
def engine_status() -> Dict[str, Any]:
    """排布引擎（nbs-fractal）状态。"""
    from core import engine as _engine

    path = _engine.find_engine()
    return {
        "installed": path is not None,
        "path": path or "",
        "expected": str(_engine.engine_path()),
        "urls": _engine.ENGINE_URLS,
    }


@app.post("/api/engine/download")
def engine_download() -> Dict[str, Any]:
    """下载排布引擎到 tools/。"""
    from core import engine as _engine

    ok, msg = _engine.download()
    return {"ok": ok, "message": msg, "installed": _engine.is_available()}


@app.get("/api/models")
def models() -> Dict[str, Any]:
    """可用分离模型 + 乐器表。"""
    return {
        "models": _load_models(),
        "instruments": [
            {"id": key, "name": name} for key, name in INSTRUMENT_NAMES.items()
        ],
    }


@app.get("/api/progress")
def progress() -> Dict[str, Any]:
    """当前任务状态快照。"""
    return JOB.snapshot()


@app.post("/api/convert")
async def convert(request: Request, background: BackgroundTasks) -> Dict[str, Any]:
    """接收上传，启动后台转换任务，立即返回 job_id。"""
    try:
        form = await request.form()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析上传表单：{exc}") from exc

    upload = form.get("file") or form.get("audio")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="缺少音频文件（字段名应为 file 或 audio）")

    # 提取普通表单字段
    params: Dict[str, Any] = {}
    for key, value in form.multi_items():
        if hasattr(value, "read"):
            continue  # 文件字段
        params[key] = value

    job_id = uuid.uuid4().hex[:12]
    if not JOB.begin(job_id):
        raise HTTPException(status_code=409, detail="已有任务在执行中，请等待当前任务完成")

    original_name = os.path.basename(getattr(upload, "filename", "") or "audio.mp3")
    safe_name = _safe_filename(os.path.splitext(original_name)[0]) or "audio"
    ext = os.path.splitext(original_name)[1] or ".mp3"
    stored_name = f"{job_id}_{safe_name}{ext}"
    upload_path = UPLOAD_DIR / stored_name
    out_dir = OUTPUT_DIR / job_id

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        with open(upload_path, "wb") as fh:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    fh.close()
                    upload_path.unlink(missing_ok=True)
                    JOB.fail(job_id, f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 限制")
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 限制",
                    )
                fh.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        JOB.fail(job_id, f"保存上传文件失败：{exc}")
        raise HTTPException(status_code=500, detail=f"保存上传文件失败：{exc}") from exc
    finally:
        try:
            await upload.close()
        except Exception:  # noqa: BLE001
            pass

    async def _background() -> None:
        await run_in_threadpool(_run_conversion, job_id, str(upload_path), str(out_dir), params)

    background.add_task(_background)

    return {"job_id": job_id, "status": "running"}


@app.get("/api/download/{kind}")
def download(kind: str) -> FileResponse:
    """下载转换结果。kind ∈ nbs | midi | litematic | instrumental。"""
    key = kind.strip().lower()
    if key not in DOWNLOAD_KINDS:
        raise HTTPException(status_code=404, detail=f"未知的文件类型：{kind}")

    with JOB._lock:  # noqa: SLF001 - 同模块内部共享锁
        path = JOB.files.get(key)
        name = JOB.names.get(key)

    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"文件尚未生成或已丢失：{kind}")

    media_type, _default_ext = DOWNLOAD_KINDS[key]
    filename = name or os.path.basename(path)
    return FileResponse(path, media_type=media_type, filename=filename)


def main() -> None:  # pragma: no cover - 供 `python web/server.py` 直接运行
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8760)


if __name__ == "__main__":  # pragma: no cover
    main()
