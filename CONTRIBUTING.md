# 贡献指南

感谢关注！这个项目是个人维护的小工具，欢迎提 issue 和 PR。

## 环境准备

```bash
git clone https://github.com/szkele1145/mp3-nbs-converter.git
cd mp3-nbs-converter
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## 模型文件

模型体积大（合计约 380 MB），没有放进 git。需要自己放到 `models/`：

| 文件 | 来源 | 用途 |
|---|---|---|
| `htdemucs_ft_vocals.onnx` | [StemSplit/demucs-onnx](https://github.com/StemSplit/demucs-onnx) | 默认模型，质量最好 |
| `htdemucs_onnx.onnx` | 同上 | 四源分离 |
| `UVR_MDXNET_Main.onnx` | [UVR5](https://github.com/Anjok07/ultimatevocalremover5) | 最省内存 |
| `nmp.onnx` | [basic-pitch](https://github.com/spotify/basic-pitch) | 音高识别 |

**Release 附件里的安装包/便携版已经内置全部模型**，直接用即可。

## 开发

```bash
python run.py                              # Web 界面
python run.py --cli song.mp3 -o ./out     # 命令行
python build_exe.py --zip                  # 打包
```

改完代码后跑一遍自测：

```bash
python work/test_edge_cases.py     # 边界输入回归
python work/verify_formats.py      # 产物格式校验
python work/scan_encoding.py       # 编码检查
```

## 提交规范

- 一次提交只做一件事，commit message 用中文说明「做了什么」
- 改动了 Python 代码请同步更新 README
- **注意编码**：所有文件必须是 UTF-8。用编辑器改，别用 PowerShell 的 `Out-File` 写中文
  （会把 UTF-8 当 GBK 读，字符会被吃掉）
- 不要提交 `dist/`、`build/`、`work/`、`models/*.onnx`

## 打包发布

```bash
python build_exe.py --zip                                   # 生成目录版 + 便携 ZIP
"C:\Users\<你>\AppData\Local\Programs\Inno Setup 6\ISCC.exe" installer.iss   # 生成安装包
```

## 许可

提交代码即表示同意以 MIT 协议开源。
