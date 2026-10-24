/* ===================================================================
   MP3 → 音符盒转换器 · 前端交互脚本
   原生 JS，无任何框架 / 外部依赖，离线可用。
   =================================================================== */
'use strict';

/* ------------------------------------------------------------------
   常量与默认数据
   ------------------------------------------------------------------ */

const API = {
  convert: '/api/convert',
  progress: '/api/progress',
  models: '/api/models',
  health: '/api/health',
  download: (kind) => `/api/download/${kind}`,
};

/** /api/models 不可用时的内置默认列表（优雅降级） */
const DEFAULT_MODELS = [
  { id: 'htdemucs', name: 'HTDemucs 四源分离（推荐，质量最好）' },
  { id: 'UVR_MDXNET_Main', name: 'UVR MDX-NET（人声分离）' },
  { id: 'htdemucs_embedded', name: 'HTDemucs 旧版导出' },
];

/** 下载卡片定义：kind 对应 /api/download/{kind} */
const DOWNLOAD_ITEMS = [
  {
    kind: 'nbs',
    icon: '🎵',
    ext: '.nbs',
    title: '音符盒文件',
    desc: 'Minecraft Note Block Studio 音乐工程，可直接用 NBS 音乐编辑器打开',
    resultKey: 'nbs',
  },
  {
    kind: 'midi',
    icon: '🎼',
    ext: '.midi',
    title: '标准 MIDI 音乐',
    desc: '通用乐谱格式，可在任意 DAW / 播放器中试听与编辑',
    resultKey: 'midi',
  },
  {
    kind: 'litematic',
    icon: '🧱',
    ext: '.litematic',
    title: 'Litematica 蓝图',
    desc: 'Minecraft 投影蓝图，按紧凑布局自动摆放音符盒与中继器',
    resultKey: 'litematic',
  },
  {
    kind: 'instrumental',
    icon: '🎧',
    ext: '.wav',
    title: '伴奏音频 WAV',
    desc: '去除人声后的纯伴奏波形，可直接试听',
    resultKey: 'instrumental',
  },
];

/** 统计信息字段的中文名与格式化方式 */
const STAT_LABELS = {
  notes: { label: '音符数', fmt: (v) => `${v}` },
  note_count: { label: '音符数', fmt: (v) => `${v}` },
  duration: { label: '时长', fmt: (v) => `${v}s` },
  duration_sec: { label: '时长', fmt: (v) => `${v}s` },
  length: { label: '时长', fmt: (v) => `${v}s` },
  tracks: { label: '轨道数', fmt: (v) => `${v}` },
  track_count: { label: '轨道数', fmt: (v) => `${v}` },
  tracks_count: { label: '轨道数', fmt: (v) => `${v}` },
  blocks: { label: '方块数', fmt: (v) => `${v}` },
  block_count: { label: '方块数', fmt: (v) => `${v}` },
  subtracks: { label: '子轨道数', fmt: (v) => `${v}` },
  columns: { label: '列数', fmt: (v) => `${v}` },
  rows: { label: '行数', fmt: (v) => `${v}` },
  tempo: { label: '速度 BPM', fmt: (v) => `${v}` },
  source: { label: '识别来源', fmt: (v) => `${v}` },
  model: { label: '分离模型', fmt: (v) => `${v}` },
  instrument: { label: '主旋律乐器', fmt: (v) => `${v}` },
  bass_instrument: { label: '低音乐器', fmt: (v) => `${v}` },
  separated: { label: '已去人声', fmt: (v) => (v ? '是' : '否') },
  separated_vocals: { label: '已去人声', fmt: (v) => (v ? '是' : '否') },
  vocal_model: { label: '分离模型', fmt: (v) => `${v}` },
  vocal_fallback: { label: '分离状态', fmt: (v) => (v ? '⚠ 已降级（效果变差）' : '正常') },
  removed_vocals: { label: '已去人声', fmt: (v) => (v ? '是' : '否') },
};

const ACCEPTED_EXT = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'];
const MAX_FILE_MB = 200;

/* ------------------------------------------------------------------
   工具函数
   ------------------------------------------------------------------ */

/** 按 id 取元素 */
const $ = (id) => document.getElementById(id);

/** 转义 HTML，避免文件名等外部文本破坏结构 */
const escapeHtml = (s) =>
  String(s ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[ch]);

/** 格式化字节数 */
const formatBytes = (bytes) => {
  if (!Number.isFinite(bytes) || bytes < 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
};

/** 格式化秒数为 mm:ss */
const formatDuration = (sec) => {
  if (!Number.isFinite(sec) || sec < 0) return '—';
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
};

/** 统一的中文错误提示 */
const friendlyError = (err) => {
  if (!err) return '发生未知错误，请稍后重试。';
  const msg = String(err.message || err);
  if (/Failed to fetch|NetworkError|load failed/i.test(msg)) {
    return '无法连接到后端服务，请确认 FastAPI 已启动（默认 http://127.0.0.1:8000）。';
  }
  if (/abort/i.test(msg)) return '请求被中断，请重试。';
  return msg;
};

/** 通用 fetch 包装：非 2xx 抛出中文友好错误 */
const request = async (url, options = {}) => {
  let res;
  try {
    res = await fetch(url, options);
  } catch (err) {
    throw new Error(friendlyError(err));
  }
  if (!res.ok) {
    let detail = '';
    try {
      detail = await res.text();
    } catch {
      detail = '';
    }
    throw new Error(`服务器返回 ${res.status}${detail ? `：${detail.slice(0, 300)}` : ''}`);
  }
  return res;
};

/* ------------------------------------------------------------------
   状态
   ------------------------------------------------------------------ */

const state = {
  file: null, // 已选择的 File
  objectUrl: null, // 本地预览用的 blob URL
  audioDuration: null, // 读取到的音频时长（秒）
  pollTimer: null, // 轮询定时器
  running: false, // 是否正在转换
  lastResult: null, // 最近一次转换结果
};

/* ------------------------------------------------------------------
   参数同步：把界面上的参数写入隐藏表单，供 FormData 提交
   ------------------------------------------------------------------ */

/**
 * 写入隐藏参数 input
 * @param {string} name 参数名
 * @param {string|number} value 参数值
 */
const setParam = (name, value) => {
  const el = document.getElementById(`p_${name}`);
  if (el) el.value = value === undefined || value === null ? '' : String(value);
};

/** 读取隐藏参数值 */
const getParam = (name) => {
  const el = document.getElementById(`p_${name}`);
  return el ? el.value : '';
};

/** 把当前界面状态整体同步到隐藏表单 */
const syncParams = () => {
  setParam('separate_vocals', $('toggle-vocal').checked ? '1' : '0');
  setParam('model', $('select-model').value);
  setParam('threads', $('number-threads').value);
  setParam('max_seconds', $('number-max-seconds').value);
  setParam('onset_threshold', Number($('number-onset').value).toFixed(2));
  setParam('min_note_length_ms', $('number-minlen').value);
  setParam('instrument', $('select-instrument').value);
  setParam('bass_instrument', $('select-bass').value);
  setParam('coarse', $('select-coarse').value);
  setParam('wrap', $('number-wrap').value);
  setParam('gap', $('number-gap').value);
  setParam('output_nbs', $('check-nbs').checked ? '1' : '0');
  setParam('output_midi', $('check-midi').checked ? '1' : '0');
  setParam('output_litematic', $('check-litematic').checked ? '1' : '0');
};

/* ------------------------------------------------------------------
   后端探测与降级
   ------------------------------------------------------------------ */

/** 检测 /api/health，失败时给出温和提示（不阻断页面） */
const checkHealth = async () => {
  const box = $('api-status');
  const text = $('api-status-text');
  try {
    const res = await request(API.health);
    const data = await res.json().catch(() => ({}));
    if (data && data.ok) {
      box.classList.add('is-ok');
      text.textContent = '后端服务已就绪';
    } else {
      box.classList.add('is-bad');
      text.textContent = '后端服务响应异常，转换可能失败';
    }
  } catch {
    box.classList.add('is-bad');
    text.textContent = '未检测到后端服务 · 界面功能仍可浏览';
  }
};

/**
 * 拉取 /api/models 填充模型下拉；接口缺失时保留内置默认列表
 */
const loadModels = async () => {
  const select = $('select-model');
  const current = select.value || 'htdemucs_ft_vocals';
  try {
    const res = await request(API.models);
    const data = await res.json().catch(() => ({}));
    const models = Array.isArray(data && data.models) ? data.models : [];
    if (!models.length) throw new Error('empty');
    select.innerHTML = models
      .map((m) => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name || m.id)}</option>`)
      .join('');
    if (models.some((m) => m.id === current)) select.value = current;
  } catch {
    // 降级：保留 HTML 中已写好的默认模型列表
    select.value = current;
  }
};

/* ------------------------------------------------------------------
   文件选择 / 拖拽 / 粘贴
   ------------------------------------------------------------------ */

/** 校验文件类型与大小 */
const validateFile = (file) => {
  if (!file) return '没有读取到文件。';
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const okType =
    ACCEPTED_EXT.includes(ext) || (file.type && file.type.startsWith('audio/'));
  if (!okType) {
    return `不支持的文件格式「${ext || '未知'}」，请选择 ${ACCEPTED_EXT.map((e) => '.' + e).join(' / ')} 文件。`;
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    return `文件过大（${formatBytes(file.size)}），请控制在 ${MAX_FILE_MB} MB 以内。`;
  }
  if (file.size === 0) return '文件内容为空，请重新选择。';
  return null;
};

/** 读取音频时长（用 <audio> 读 metadata） */
const readDuration = (file) =>
  new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    const done = (value) => {
      URL.revokeObjectURL(url);
      resolve(value);
    };
    // 兜底：某些环境读不到 metadata，5 秒后按未知处理
    const timer = setTimeout(() => done(null), 5000);
    audio.addEventListener('loadedmetadata', () => {
      clearTimeout(timer);
      done(Number.isFinite(audio.duration) ? audio.duration : null);
    });
    audio.addEventListener('error', () => {
      clearTimeout(timer);
      done(null);
    });
    audio.preload = 'metadata';
    audio.src = url;
  });

/** 选中文件后的统一处理 */
const handleFile = async (file) => {
  const err = validateFile(file);
  if (err) {
    showError(err);
    return;
  }
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.audioDuration = null;

  $('file-name').textContent = file.name;
  $('file-sub').textContent = `${formatBytes(file.size)} · 正在读取时长…`;
  $('file-card').classList.remove('hidden');
  $('file-input').value = '';
  $('action-note').textContent = '参数已就绪，点击下方按钮开始转换';
  updateConvertButton();

  const duration = await readDuration(file);
  if (state.file !== file) return; // 期间用户又换了文件
  state.audioDuration = duration;
  $('file-sub').textContent = `${formatBytes(file.size)} · 时长 ${formatDuration(duration)}`;

  // 本地预览音频
  state.objectUrl = URL.createObjectURL(file);
  $('audio-original').src = state.objectUrl;
};

/** 清空当前文件 */
const clearFile = () => {
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
  state.file = null;
  state.audioDuration = null;
  $('file-card').classList.add('hidden');
  $('file-name').textContent = '—';
  $('file-sub').textContent = '—';
  $('file-input').value = '';
  $('audio-original').removeAttribute('src');
  $('audio-original').load();
  $('action-note').textContent = '请先选择一个音频文件';
  updateConvertButton();
};

/** 根据是否已有文件更新按钮可用状态 */
const updateConvertButton = () => {
  const btn = $('btn-convert');
  btn.disabled = state.running || !state.file;
  if (state.running) {
    btn.classList.add('is-busy');
    $('btn-label').textContent = '转换中…';
  } else {
    btn.classList.remove('is-busy');
    $('btn-label').textContent = '开始转换';
  }
};

/* ------------------------------------------------------------------
   转换流程
   ------------------------------------------------------------------ */

/** 提交转换任务 */
const startConvert = async () => {
  if (!state.file || state.running) return;
  hideError();
  syncParams();

  const form = new FormData();
  form.append('file', state.file, state.file.name);
  // 把所有隐藏参数一并附加
  form.append('separate_vocals', getParam('separate_vocals'));
  form.append('model', getParam('model'));
  form.append('threads', getParam('threads'));
  form.append('max_seconds', getParam('max_seconds'));
  form.append('onset_threshold', getParam('onset_threshold'));
  form.append('min_note_length_ms', getParam('min_note_length_ms'));
  form.append('instrument', getParam('instrument'));
  form.append('bass_instrument', getParam('bass_instrument'));
  form.append('coarse', getParam('coarse'));
  form.append('wrap', getParam('wrap'));
  form.append('gap', getParam('gap'));
  form.append('output_nbs', getParam('output_nbs'));
  form.append('output_midi', getParam('output_midi'));
  form.append('output_litematic', getParam('output_litematic'));

  state.running = true;
  updateConvertButton();
  $('action-note').textContent = '正在上传音频并启动转换任务…';
  showProgress(0, '准备中…', '正在上传文件…');
  $('result-section').classList.add('hidden');

  try {
    const res = await request(API.convert, { method: 'POST', body: form });
    const data = await res.json().catch(() => ({}));
    if (!data || !data.job_id) {
      throw new Error('后端未返回 job_id，任务可能未成功启动。');
    }
    $('action-note').textContent = `任务已启动（ID: ${data.job_id}）`;
    startPolling();
  } catch (err) {
    state.running = false;
    updateConvertButton();
    $('action-note').textContent = '转换未启动，请检查后端服务';
    showError(friendlyError(err));
  }
};

/** 轮询进度 */
const startPolling = () => {
  stopPolling();
  // 立刻查一次，避免等待 1 秒的空窗
  pollProgress();
  state.pollTimer = setInterval(pollProgress, 1000);
};

const stopPolling = () => {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
};

const pollProgress = async () => {
  try {
    const res = await request(API.progress);
    const data = await res.json().catch(() => ({}));
    const status = String((data && data.status) || 'idle');
    const progress = Number.isFinite(Number(data && data.progress))
      ? Math.max(0, Math.min(1, Number(data.progress)))
      : 0;

    if (status === 'done') {
      stopPolling();
      state.running = false;
      updateConvertButton();
      showProgress(1, '转换完成', (data && data.message) || '全部文件已生成');
      renderResult((data && data.result) || {});
      $('action-note').textContent = '转换完成，可以下载文件了';
      return;
    }

    if (status === 'error') {
      stopPolling();
      state.running = false;
      updateConvertButton();
      showProgress(progress, '转换失败', (data && data.message) || '');
      showError((data && data.error) || '转换失败，请查看后端日志。');
      $('action-note').textContent = '转换失败，请调整参数后重试';
      return;
    }

    const stage = (data && data.stage) || (status === 'running' ? '处理中…' : '等待中…');
    let message = (data && data.message) || '';
    // 显示预计剩余时间
    const eta = data && data.eta;
    if (status === 'running' && typeof eta === 'number' && eta > 0 && message.indexOf('预计剩余') === -1) {
      let etaText;
      if (eta < 60) {
        etaText = '预计剩余 ' + Math.max(1, Math.round(eta)) + 's';
      } else {
        const m = Math.floor(eta / 60);
        const s = Math.round(eta % 60);
        etaText = '预计剩余 ' + m + 'm' + String(s).padStart(2, '0') + 's';
      }
      if (message.indexOf(' · ') !== -1) {
        message = message + ' · ' + etaText;
      } else {
        message = stage + ' · ' + Math.round(progress * 100) + '% · ' + etaText;
      }
    }
    showProgress(progress, stage, message);
  } catch (err) {
    // 单次轮询失败不立即中断：连续失败由下面的容忍计数处理
    state.pollFailures = (state.pollFailures || 0) + 1;
    if (state.pollFailures >= 5) {
      stopPolling();
      state.running = false;
      updateConvertButton();
      showError(friendlyError(err));
      $('action-note').textContent = '与后端失去联系，轮询已停止';
    }
  }
};

/* ------------------------------------------------------------------
   界面更新
   ------------------------------------------------------------------ */

/** 更新进度条 */
const showProgress = (ratio, stage, message) => {
  $('progress-card').classList.remove('hidden');
  const pct = Math.round(ratio * 100);
  $('progress-fill').style.width = `${pct}%`;
  $('progress-percent').textContent = `${pct}%`;
  $('progress-stage').textContent = stage || '处理中…';
  $('progress-message').textContent = message || '';
  $('progress-bar').setAttribute('aria-valuenow', String(pct));
};

/** 显示错误卡片 */
const showError = (message) => {
  $('error-text').textContent = message;
  $('error-card').classList.remove('hidden');
};

/** 隐藏错误卡片 */
const hideError = () => {
  $('error-card').classList.add('hidden');
  $('inline-error-text').classList.add('hidden');
};

/** 渲染下载卡片与统计信息 */
const renderResult = (result) => {
  state.lastResult = result || {};

  // 下载卡片
  const grid = $('download-grid');
  const sizes = state.lastResult.sizes || {};
  grid.innerHTML = DOWNLOAD_ITEMS.map((item) => {
    const fileName = state.lastResult[item.resultKey] || '';
    const shown = fileName ? String(fileName).split(/[\\/]/).pop() : '未生成';
    const size = sizes[item.kind];
    const sizeText = typeof size === 'number' && size > 0 ? formatBytes(size) : '';
    return `
      <div class="dl-card">
        <div class="dl-icon" aria-hidden="true">${item.icon}</div>
        <div class="dl-body">
          <div class="dl-title">${escapeHtml(item.title)}<span class="dl-ext">${escapeHtml(item.ext)}</span></div>
          <div class="dl-desc">${escapeHtml(item.desc)}</div>
          <div class="dl-name" title="${escapeHtml(shown)}">${escapeHtml(shown)}${sizeText ? ' · ' + escapeHtml(sizeText) : ''}</div>
        </div>
        <a class="btn dl-btn" href="${escapeHtml(API.download(item.kind))}" download>下载</a>
      </div>`;
  }).join('');

  // 统计信息
  const stats = state.lastResult.stats || {};
  const statGrid = $('stats-grid');
  const entries = Object.entries(stats).filter(([, v]) => v !== null && v !== undefined && v !== '');
  if (entries.length) {
    statGrid.innerHTML = entries
      .map(([key, value]) => {
        const meta = STAT_LABELS[key] || { label: key, fmt: (v) => `${v}` };
        return `
          <div class="stat-item">
            <div class="stat-label">${escapeHtml(meta.label)}</div>
            <div class="stat-value">${escapeHtml(meta.fmt(value))}</div>
          </div>`;
      })
      .join('');
    $('stats-box').classList.remove('hidden');
  } else {
    statGrid.innerHTML = '<div class="stat-item"><div class="stat-label">暂无统计数据</div></div>';
    $('stats-box').classList.remove('hidden');
  }

  // 伴奏试听
  const instrumental = $('audio-instrumental');
  instrumental.src = API.download('instrumental');
  instrumental.load();

  $('result-section').classList.remove('hidden');
  $('result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
};

/* ------------------------------------------------------------------
   事件绑定
   ------------------------------------------------------------------ */

const bindEvents = () => {
  const dropzone = $('dropzone');
  const fileInput = $('file-input');

  // 点击选择
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) handleFile(file);
  });

  // 拖拽
  ['dragenter', 'dragover'].forEach((type) => {
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'dragend'].forEach((type) => {
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove('dragover');
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length) handleFile(files[0]);
  });

  // 防止在页面其他区域拖放时浏览器直接打开文件
  ['dragover', 'drop'].forEach((type) => {
    window.addEventListener(type, (e) => {
      if (e.target && e.target.closest && e.target.closest('#dropzone')) return;
      e.preventDefault();
    });
  });

  // 粘贴
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData && e.clipboardData.files;
    if (items && items.length) {
      e.preventDefault();
      handleFile(items[0]);
    }
  });

  // 移除文件
  $('remove-file').addEventListener('click', clearFile);

  // 智能去人声开关文案
  const vocalToggle = $('toggle-vocal');
  const vocalText = $('vocal-switch-text');
  const updateVocalText = () => {
    vocalText.textContent = vocalToggle.checked
      ? '已开启 · 转换前先做人声分离，只保留伴奏'
      : '已关闭 · 将直接对原始音频做音高识别';
  };
  vocalToggle.addEventListener('change', updateVocalText);
  updateVocalText();

  // 模型说明随选择变化
  const MODEL_NOTES = {
    htdemucs_ft_vocals: '人声专家模型：去人声最干净，转换最快；内存占用约 4.6 GB。',
    htdemucs: '四源分离：可同时拿到鼓/贝斯，速度快；内存约 4.6 GB。',
    UVR_MDXNET_Main: '最省内存：只占约 1.5 GB，适合边打游戏边转换；比 HTDemucs 慢约 4 倍。',
  };
  const modelSelect = $('select-model');
  const modelHintEl = $('model-hint');
  const paintModelHint = () => {
    if (modelHintEl) modelHintEl.textContent = MODEL_NOTES[modelSelect.value] || '';
  };
  modelSelect.addEventListener('change', paintModelHint);
  paintModelHint();

  // 🎮 游戏模式：一键切到省内存模型 + 低线程
  const gameBtn = $('btn-game-mode');
  if (gameBtn) {
    gameBtn.addEventListener('click', () => {
      const on = gameBtn.classList.toggle('is-on');
      modelSelect.value = on ? 'UVR_MDXNET_Main' : 'htdemucs_ft_vocals';
      const threads = on ? '2' : '0';
      $('range-threads').value = threads;
      $('number-threads').value = threads;
      gameBtn.textContent = on ? '🎮 游戏模式 · 已开启' : '🎮 游戏模式';
      paintModelHint();
      $('range-threads').dispatchEvent(new Event('input'));
      syncParams();
    });
  }

  // 滑块 ↔ 数字输入联动
  const bindSliderNumber = (rangeId, numberId, hintId, decimals, formatter) => {
    const range = $(rangeId);
    const number = $(numberId);
    const hint = hintId ? $(hintId) : null;
    const fmt = formatter || ((v) => Number(v).toFixed(decimals));
    const paint = () => {
      const min = Number(range.min);
      const max = Number(range.max);
      const pct = ((Number(range.value) - min) / (max - min)) * 100;
      range.style.setProperty('--fill', `${Math.max(0, Math.min(100, pct))}%`);
      if (hint) hint.textContent = fmt(range.value);
    };
    range.addEventListener('input', () => {
      number.value = range.value;
      paint();
      syncParams();
    });
    number.addEventListener('input', () => {
      const v = Number(number.value);
      if (!Number.isFinite(v)) return;
      const min = Number(range.min);
      const max = Number(range.max);
      const clamped = Math.max(min, Math.min(max, v));
      range.value = String(clamped);
      paint();
      syncParams();
    });
    paint();
  };
  bindSliderNumber('range-onset', 'number-onset', 'hint-onset', 2);
  bindSliderNumber('range-minlen', 'number-minlen', 'hint-minlen', 1);
  bindSliderNumber('range-threads', 'number-threads', 'hint-threads', 0, (v) =>
    Number(v) === 0 ? '自动' : `${v} 线程`);

  // 其它参数变化时同步
  ['select-model', 'select-instrument', 'select-bass', 'select-coarse', 'number-wrap', 'number-gap',
    'check-nbs', 'check-midi', 'check-litematic', 'number-threads', 'number-max-seconds'].forEach((id) => {
    const el = $(id);
    if (el) el.addEventListener('change', syncParams);
  });

  // 转换按钮
  $('btn-convert').addEventListener('click', startConvert);

  // 重试
  $('btn-retry').addEventListener('click', () => {
    hideError();
    if (state.file) startConvert();
  });

  // 离开页面前清理 blob URL 与定时器
  window.addEventListener('beforeunload', () => {
    stopPolling();
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  });
};

/* ------------------------------------------------------------------
   初始化
   ------------------------------------------------------------------ */

const init = () => {
  bindEvents();
  syncParams();
  updateConvertButton();
  checkHealth();
  loadModels();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
