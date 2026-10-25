# 🎵 MP3 → 音符盒转换器

[![Release](https://img.shields.io/github/v/release/szkele1145/mp3-nbs-converter?label=Release&color=blue)](https://github.com/szkele1145/mp3-nbs-converter/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](installer.iss)

一键把歌曲（MP3 / WAV / M4A / FLAC / OGG / AAC）转成 **Minecraft 音符盒**作品：

- **`.nbs`** — Note Block Studio 歌曲文件，可直接导入 [u-u.life/nbs 在线编辑器](https://u-u.life/nbs/#compact)
- **`.midi`** — 标准 MIDI 文件，导入任何 DAW 或音符盒工具
- **`.litematic`** — Litematica 蓝图，游戏内直接放置音符盒阵列
- **伴奏 `.wav`** — 智能去人声后的纯伴奏音频

## 📥 下载

去 **[Releases 页面](https://github.com/szkele1145/mp3-nbs-converter/releases/latest)** 下载：

| 附件 | 说明 |
|---|---|
| `MP3NBS-Setup.exe` | Windows 安装包，双击安装，**启动约 1 秒** |
| `MP3NBS-Portable.zip` | 免安装版，解压即用 |

> 附件里已经内置全部模型（约 380 MB），**不用另外下载模型**。
> 想换模型？把新的 `.onnx` 放到 exe 同级的 `models/` 目录即可。

## ⚡ 资源占用（重要）

**模型选择直接决定内存占用和速度**（实测数据，12 核机器 / 4 线程）：

| 模型 | 伴奏残留 rms | 内存峰值 | 30s 耗时 | 速度 | 说明 |
|---|---|---|---|---|---|
| `htdemucs_ft_vocals` | **0.00094** | 4668 MB | 7.6s | **4x 实时** | 质量最好、也最快（默认） |
| `htdemucs` | 0.00180 | 4667 MB | 6.3s | **4.7x 实时** | 四源分离，另可拿到鼓/贝斯 |
| `UVR_MDXNET_Main` | 0.00390 | **1483 MB** | 35.9s | 0.8x 实时 | **最省内存，但慢 4~5 倍** |

> **两个选择**：
> - **赶时间 / 质量优先** → 用默认的 `htdemucs_ft_vocals`，又快又干净，只是吃 4.6 GB 内存。
> - **边打游戏边转** → 切到 `UVR_MDXNET_Main`，内存从 4.6 GB 降到 1.5 GB，
>   但要接受慢 4~5 倍（30 秒音频约 36 秒，一首 4 分钟的歌约 8 分钟）。
>
> MDX-NET 慢的原因：TDFN 网络在 3072 点频率轴上做 2D 卷积，是**计算密集型**负载。
> 批处理、线程数、重叠率都实测过，吞吐恒定在 0.8x 实时，CPU 上无法根本优化。

默认已做以下优化，尽量不打扰你打游戏：

| 措施 | 效果 |
|---|---|
| 推理线程限制为「核心数 ÷ 3」 | 12 核机器只用 4 线程，留 8 核给游戏 |
| 转换时进程优先级降到「低于正常」 | 游戏优先调度，转换只插空跑 |
| 关闭 ONNX CPU 内存竞技场 | 峰值内存省约 1 GB（3.9 → 2.9 GB 私有内存） |
| 分块折叠缓冲（只留伴奏+人声两路）| 比存 4 源省一半 |
| **去人声结果自动缓存** | 同一首歌调音高参数重转：**32 秒 → 1 秒** |
| **转换结束立刻释放模型** | 内存从峰值 4.6 GB 回落到 ~150 MB |

命令行可以进一步调：

```bash
# 只用 2 线程，最温和
python run.py --cli song.mp3 -o ./out --threads 2

# 省内存模型（1.5 GB）
python run.py --cli song.mp3 -o ./out --model UVR_MDXNET_Main

# 想快点、不在乎占用
python run.py --cli song.mp3 -o ./out --threads 8 --high-memory --normal-priority
```

**建议**：转换时先设 30 秒试听效果，满意后再转全曲。

### 去人声缓存

分离结果会按「音频内容 + 模型 + 截断长度」缓存到 `work/cache/`。
**同一首歌改起音阈值、乐器、中继器粒度等参数重转时，会直接复用上次的伴奏**，
耗时从 30 秒级降到 1 秒级。缓存只保留最近 12 个文件，超过 200 MB 的单曲不缓存。

## ✨ 特性

### 智能去人声（只留伴奏）
三个可选模型，界面里直接标注每个模型的内存占用：

- **`htdemucs_ft_vocals`**（默认）— HTDemucs-FT 人声专家，伴奏中人声残留 rms 0.00094，4x 实时
- **`htdemucs`** — 四源分离（drums / bass / other / vocals），另可拿到鼓与贝斯，4.7x 实时
- **`UVR_MDXNET_Main`** — UVR MDX-NET，内存只要 1.5 GB，但慢 4~5 倍，适合边打游戏边转

界面右上角有 **🎮 游戏模式** 按钮：一键切到 `UVR_MDXNET_Main` + 2 线程，
直接把内存占用从 4.6 GB 压到 1.5 GB。再点一次恢复默认。

都不可用时自动退化为频谱中频抑制法，功能不中断；界面会用 ⚠ 标出「已降级」，
降级结果也**不写入缓存**（模型修好后能重新算）。

### 音高识别
移植 **Spotify basic-pitch**（ONNX 版 `nmp.onnx`），后处理严格对齐官方实现：
帧概率 → 音符拼接 → 起音/帧双重阈值 → 最短音符过滤。

### Litematica 蓝图

按 **Litematica v6** 格式（gzip 压缩 NBT）写出 `.litematic`，游戏里 `//litematica` 一键加载。

**排布引擎（推荐）**

默认调用 [nbs-fractal](https://github.com/omninbs/nbs-fractal) 生成投影，**结果与
[u-u.life/nbs](https://u-u.life/nbs/#compact) 完全一致**。引擎不随安装包分发
（AGPL-3.0），而是按需获取：

**方式一：一键下载**

界面里勾选「输出 .litematic」时若检测到缺失，会弹出提示并给下载按钮，
文件存到 exe 同级的 `tools/nbs-fractal.exe`。

**方式二：自己编译（最可靠）**

`nbs-fractal` 已发布到 [crates.io](https://crates.io/crates/nbs-fractal)，
装了 Rust 后一条命令就能编：

```bash
# 1. 安装 Rust
#    官方源在国内常常连不上，用中科大镜像：
#    先下载 rustup-init.exe：
#      https://mirrors.ustc.edu.cn/rust-static/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe
#    然后设置环境变量再运行：
set RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
rustup-init.exe -y --default-toolchain stable --profile minimal

# 2. 配置 crates 镜像（避免 crates.io 直连失败）
#    在 %USERPROFILE%\.cargo\config.toml 里写：
#      [source.crates-io]
#      replace-with = 'ustc'
#      [source.ustc]
#      registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"

# 3. 编译并安装到 tools/
cargo install nbs-fractal --version ">=0.1.2, <0.2.0" --root ./tools

# 4. 编好的程序在 tools/bin/nbs-fractal.exe
move tools\bin\nbs-fractal.exe tools\
```

编译需要 MSVC 构建工具（Visual Studio Build Tools 的「使用 C++ 的桌面开发」）。
没有的话 rustup 会提示 `installing msvc toolchain without its prerequisites`。

**方式三：手动放置**

有别人给你的 `nbs-fractal.exe`，直接丢进 `tools/` 目录就行。

**内置排布（备选）**

引擎不可用时软件会**询问**你是否改用内置排布：

> ⚠️ 内置排布是参考 nbs-fractal 思路的**简化实现**，效果较差、格式不稳定，
> 形状与官方输出**不一致**。除非你明确接受，否则建议先装好引擎。

命令行下：不加 `--builtin-layout` 时，缺引擎会直接报错；加了这个开关才用内置排布。

**参数**

`wrap`（每行最大列数）/ `gap`（音轨间距）/ `coarse`（中继器粒度）三个参数
语义与 nbs-fractal CLI 保持一致，用引擎时行为相同。

### Web 界面
FastAPI + 原生 JS 单页应用：
- 拖拽 / 粘贴 / 点击上传
- 实时进度（分阶段：解码 → 去人声 → 识别 → 写文件）
- **预计剩余时间（ETA）**——分离阶段也按分块实时更新，不会卡在 8%
- 原曲 vs 伴奏双通道试听
- 一键下载全部四种产物
- 内置 [u-u.life/nbs 紧凑布局编辑器](https://u-u.life/nbs/#compact) 快捷入口

## 🚀 快速开始

### 方式一：安装包（推荐）

1. 从 [Releases](https://github.com/szkele1145/mp3-nbs-converter/releases/latest) 下载 `MP3NBS-Setup.exe`
2. 双击安装（中文界面），装好后从开始菜单或桌面启动
3. 浏览器自动打开 <http://127.0.0.1:8760>，拖入 MP3，点「开始转换」

安装版是目录形式，**启动约 1 秒**。

### 方式二：便携版（免安装）

1. 下载 `MP3NBS-Portable.zip` 并解压
2. 双击 `MP3NBS转换器.exe`
3. 浏览器打开 <http://127.0.0.1:8760>

### 方式三：源码运行

```bash
# Python 3.10+（开发环境 3.14）
git clone https://github.com/szkele1145/mp3-nbs-converter.git
cd mp3-nbs-converter
pip install -r requirements.txt

# Web 界面
python run.py

# 命令行转换
python run.py --cli song.mp3 -o ./out
```

> 源码运行时需要自己准备模型文件，见 [CONTRIBUTING.md](CONTRIBUTING.md)。
> Release 附件里已经内置全部模型，不用管这些。

## 🖥️ 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cli <文件>` | — | 命令行模式：转换指定音频 |
| `-o, --output <目录>` | `output` | 输出目录 |
| `--no-vocals` | 关 | 不去人声，直接用原始音频 |
| `--model <id>` | `htdemucs` | 分离模型：`htdemucs` / `UVR_MDXNET_Main` |
| `--onset <0-1>` | `0.5` | 起音阈值，越大漏检越多、音符越稀 |
| `--frame <0-1>` | `0.3` | 帧阈值 |
| `--min-note-ms <毫秒>` | `127.7` | 音符最短时长 |
| `--instrument <名>` | `AUTO` | 主旋律乐器，AUTO = 按音域自动匹配 |
| `--bass-instrument <名>` | `AUTO` | 低音轨乐器，AUTO = 按音域自动匹配 |
| `--wrap <列数>` | `0` | 每行最大列数，0 = 不限 |
| `--gap <格>` | `0` | 轨道间距 |
| `--max-seconds <秒>` | `0` | 只转换前 N 秒，0 = 不限 |
| `--no-nbs` / `--no-midi` / `--no-litematic` | — | 跳过对应产物 |
| `--auto-preset` | 关 | 懒人模式：按音频自动挑参数 |
| `--builtin-layout` | 关 | 缺排布引擎时允许用内置实现（效果较差） |
| `--host` / `--port` | `127.0.0.1:8760` | Web 服务监听地址 |
| `--no-browser` | 关 | 启动服务时不自动打开浏览器 |

示例：

```bash
# 去人声 + 自定义乐器 + 限制 60 秒
python run.py --cli song.mp3 -o ./out --instrument FLUTE --max-seconds 60

# 不去人声，转钢琴谱
python run.py --cli song.mp3 -o ./out --no-vocals --instrument PIANO
```

## 📦 产物说明

| 文件 | 用途 |
|---|---|
| `曲名.nbs` | 导入 [NBS](https://u-u.life/nbs/#compact) 或游戏内 Note Block Studio |
| `曲名.midi` | 任何 MIDI 播放器 / DAW |
| `曲名.litematic` | Litematica 模组：`Ctrl+M` → Load Schematics |
| `曲名_instrumental.wav` | 去人声后的伴奏 |

`.nbs` / `.midi` 是**时间轴音轨**（音符按播放时序排布，导入后直接能播）；
`.litematic` 是**空间紧凑布局**（音符盒 + 中继器阵列，游戏内直接摆好）。
`wrap` / `gap` / `coarse` 三个排布参数只影响 `.litematic`。

## 🗂️ 项目结构

```
mp3-nbs-converter/
├── run.py                  # 入口（CLI + Web）
├── build_exe.py            # PyInstaller 打包脚本
├── requirements.txt
├── core/
│   ├── audio_io.py         # ffmpeg 解码、重采样、WAV 写出
│   ├── transcribe.py       # basic-pitch ONNX 移植（音高识别）
│   ├── vocal_separate.py   # HTDemucs / MDX-NET 人声分离
│   ├── nbs_writer.py       # NBS v5 写出（含同 tick 同层冲突消解）
│   ├── midi_writer.py      # MIDI 写出
│   ├── compact_layout.py   # 音符盒阵列排布 + Litematica 蓝图写出
│   ├── nbt.py              # Minecraft NBT 序列化
│   ├── pipeline.py         # 端到端流水线
│   └── paths.py            # 源码/打包双模式路径解析
├── web/
│   ├── server.py           # FastAPI（上传/进度/下载/模型列表）
│   ├── index.html
│   ├── app.js
│   └── style.css
├── models/                 # ONNX 模型（打包时收进 exe）
│   ├── htdemucs_ft_vocals.onnx  # HTDemucs-FT 人声专家（158 MB，默认）
│   ├── htdemucs_onnx.onnx      # HTDemucs 四源分离（158 MB）
│   ├── UVR_MDXNET_Main.onnx    # UVR MDX-NET（64 MB，最省内存）
│   └── nmp.onnx                # basic-pitch 音高识别（0.2 MB）
└── work/                   # 运行时工作目录（上传/输出/缓存）
```

## 🔧 技术细节

### NBS v5 写出
- 字符串字段是 **cp1252**（pynbs 硬编码），中文无法写入 → 描述字段用英文，
  中文曲名会被丢弃（不影响音符内容）
- `new_file()` 预建的第 0 层会被清掉，层列表由实际轨道重建
- **同一 tick 同一层的音符会冲突**（层跳 0 会截断层列表），按 (tick, layer)
  分配槽位，冲突时自动开新层
- 音符写入前按 tick 排序
- `tempo` 字段语义是「每秒 tick 数」（标准 10），不是 BPM

### MIDI
- SMF type 1，meta 文本 latin-1 消毒
- `message.time` 是**绝对秒**（mido 自动转 delta），全部 `int(round(x*1000))` 微秒化

### Litematic
- NBT 复合标签以 `TAG_END` 结尾；时间戳用 `TAG_LONG`（int32 会溢出）
- 方块索引 = `x + y*size_x*size_z + z*size_x`，palette 索引 0 = air
- 位包 little-endian，long 内也小端
- Litematic 走 NBT，**支持 UTF-8 中文**（与 NBS 不同）

### UVR MDX-NET 分离
- 模型 I/O = `(batch, 4, 3072, 256)`，4 通道 = `[L.real, L.imag, R.real, R.imag]`
- 输入是**原始 STFT**（不压缩、不归一化），输出直接是人声复数谱
- `n_fft=6144`，`hop=1024`，center=True，periodic Hann
- 分块 25% 重叠（step = `dim_t * 3 // 4`）

### HTDemucs 分块
- 块长 343980（约 7.8 秒 @44.1 kHz），25% 重叠，hann 过渡窗
- 峰值归一化后再推理，避免数值溢出
- 源顺序固定 **drums / bass / other / vocals**；4 源之和恒等于原混音
- `htdemucs_ft_vocals` 是**人声专家模型**，伴奏要取 `mix - vocals`，
  不能用其他 3 源求和

## ❓ FAQ

**Q: 转换很慢？**
A: 人声分离在 CPU 上约 0.6x 实时（一首 4 分钟歌约 6 分钟）。先在界面里把
「只转换前 N 秒」设小一点试试效果，满意后再转全曲。

**Q: 去人声后有残留？**
A: HTDemucs 对强混响/和声密集的歌会有少量残留。可以：
- 换 `UVR_MDXNET_Main` 模型对比效果
- 界面里提高起音阈值，减少误识别的「人声残影」

**Q: 生成的 .nbs 打不开？**
A: 确认用的是 NBS 2.x 或 [u-u.life/nbs](https://u-u.life/nbs/)（v5 格式）。
NBS 的字符串字段是 cp1252，**中文曲名无法写入**（会被丢弃），建议用英文曲名。

**Q: exe 启动很慢？**
A: 用**安装包版或便携 ZIP**（都是目录版，启动约 1 秒）。
单文件 exe 每次启动要解包约 520 MB 到临时目录，实测冷启动 2-3 分钟，**本项目不提供**。

**Q: 转换时内存占用太高影响游戏？**
A: 切到 **🎮 游戏模式**（`UVR_MDXNET_Main` + 2 线程），内存从 4.6 GB 降到 1.5 GB。
代价是速度慢 4~5 倍。转换结束模型会立即释放，服务常驻内存约 110 MB。

**Q: 转换太慢？**
A: 确认用的是默认的 `htdemucs_ft_vocals`（4x 实时），而不是 `UVR_MDXNET_Main`（0.8x 实时）。
另外用「只转换前 N 秒」先试听，满意再转全曲；同一首歌改参数重转会命中缓存，几乎瞬间完成。

**Q: 能离线用吗？**
A: 能。全部模型本地推理，不联网。

## 📄 许可与致谢

| 组件 | 来源 | 许可 |
|---|---|---|
| HTDemucs v4 | [facebookresearch/demucs](https://github.com/facebookresearch/demucs)（MIT），ONNX 导出参考 [StemSplit/demucs-onnx](https://github.com/StemSplit/demucs-onnx) | MIT |
| basic-pitch | [spotify/basic-pitch](https://github.com/spotify/basic-pitch) | Apache-2.0 |
| UVR MDX-NET | [UVR5](https://github.com/Anjok07/ultimatevocalremover5) 模型权重 | MIT（代码）/ 模型遵循其声明 |
| nbs-fractal | 排布算法参考（简化实现，非移植） | AGPL-3.0 |
| rustmatica | Litematica 格式参考 | MIT |
| NBS 格式 | [Note Block Studio](https://opennbs.org/) / [pynbs](https://github.com/OpenNBS/pynbs) | MIT |

本项目代码 MIT 许可。模型权重各自遵循上游许可，商用请自行确认。
