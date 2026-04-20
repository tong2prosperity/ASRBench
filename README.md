# ASRBench

中文语音识别（ASR）基准测试框架。在 [Common Voice](https://commonvoice.mozilla.org/) zh-CN 语料上对多家云端 ASR 服务进行评测，输出 CER / WER / RTF 指标。

支持的 ASR 服务：

| Provider | 引擎 | 接入方式 |
|---|---|---|
| **VolcEngine** | 火山引擎大模型语音识别 | WebSocket (bigmodel v3) |
| **FunASR** | 阿里 DashScope FunASR | REST (dashscope SDK) |
| **Qwen ASR** | 通义千问 qwen3-asr | WebSocket (dashscope OmniRealtime) |
| **Paraformer** | 阿里 Paraformer v2 | REST (dashscope SDK) |
| **Tencent** | 腾讯云语音识别 | WebSocket (streaming SDK) |

## 环境准备

需要 Python >= 3.10 和 [uv](https://docs.astral.sh/uv/)。

```bash
# 安装 uv（如果没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/tong2prosperity/ASRBench.git
cd ASRBench

# 安装依赖（自动创建 .venv）
uv sync
```

## 准备语料

ASRBench 默认使用 Mozilla Common Voice zh-CN 语料。你需要自行下载并放置到 `corpus/zh-CN/` 目录。

### Common Voice zh-CN

1. 前往 [Common Voice](https://commonvoice.mozilla.org/zh-CN/datasets) 下载 zh-CN 数据集
2. 解压后将内容放置为如下结构：

```
corpus/
  zh-CN/
    test.tsv
    dev.tsv
    clips/
      common_voice_zh-CN_xxxxx.mp3
      ...
```

TSV 文件中需要包含 `path` 和 `sentence` 两列，`clips/` 目录下存放对应的音频文件。

### ASCEND MIXED（可选）

仓库附带了一个下载脚本，可以从 Hugging Face 下载 ASCEND_MIXED 数据集：

```bash
uv run python scripts/download_ascend_mixed.py
```

下载完成后语料位于 `corpus/ascend-mixed/`，使用时需指定配置文件中的 `corpus_dir`。

## 配置

将配置模板复制为 `config.yaml`，填入你的 API 密钥：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，将各 provider 的占位符替换为真实凭据：

```yaml
providers:
  volcengine:
    enabled: true
    app_key: "你的火山引擎 App Key"
    access_key: "你的火山引擎 Access Key"
    ...
  funasr:
    enabled: true
    api_key: "你的 DashScope API Key"
    ...
  tencent:
    enabled: true
    app_id: "你的腾讯云 App ID"
    secret_id: "你的腾讯云 Secret ID"
    secret_key: "你的腾讯云 Secret Key"
    ...
```

**凭据获取方式：**

- **VolcEngine** — [火山引擎控制台](https://console.volcengine.com/) → 语音技术 → 获取 App Key / Access Key
- **FunASR / Qwen ASR / Paraformer** — [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/) → API Key 管理，三者共用同一个 API Key
- **Tencent** — [腾讯云控制台](https://console.cloud.tencent.com/) → 访问管理 → API 密钥管理

不需要的 provider 可以设置 `enabled: false` 跳过。

## 运行

```bash
# 运行所有已启用的 provider
uv run python -m asrbench

# 只运行单个 provider
uv run python -m asrbench --provider volcengine

# 指定数据集分片
uv run python -m asrbench --dataset dev        # dev / test / both

# 限制样本数（用于快速验证）
uv run python -m asrbench --limit 50

# 调整并发数
uv run python -m asrbench --concurrency 20

# 断点续跑（跳过已有结果的样本）
uv run python -m asrbench --resume

# 快速冒烟测试
uv run python -m asrbench --provider volcengine --limit 1 --concurrency 1
```

结果保存在 `results/<provider>.json`。

## 生成报告

跑完 benchmark 后，生成 Markdown 格式的评测报告：

```bash
uv run python -m asrbench.generate_report
```

报告输出到 `results/benchmark_report.md`，同时生成 `results/benchmark_summary.json`。

可选参数：

```bash
uv run python -m asrbench.generate_report --results-dir results --output report.md
```

## 指标说明

| 指标 | 含义 |
|---|---|
| **CER** | 字符错误率（corpus 级）— `total_edits / total_reference_chars`，按字符拆分计算 |
| **WER** | 词错误率（corpus 级）— 使用 jieba 分词后计算 |
| **RTF** | 实时率 — `识别耗时 / 音频时长`，越低越快 |

文本在计算前统一做 NFKC 归一化 + 去除中英文标点 + 小写化处理。

## 项目结构

```
asrbench/
  __main__.py          # 入口
  runner.py            # 异步评测调度器
  providers/           # ASR 服务适配层
    base.py            #   抽象基类
    volcengine.py      #   火山引擎
    funasr.py          #   FunASR
    qwen_asr.py        #   Qwen ASR
    paraformer.py      #   Paraformer
    tencent.py         #   腾讯云
  audio.py             # 音频加载与格式转换
  dataset.py           # 语料加载
  metrics.py           # CER / WER 计算
  report.py            # 运行时结果保存与摘要
  generate_report.py   # 独立报告生成器
vendor/
  tencentcloud_speech_sdk/  # 腾讯云流式语音 SDK（PyPI 未发布，本地 vendored）
scripts/
  download_ascend_mixed.py  # ASCEND MIXED 语料下载脚本
config.yaml.example         # 配置模板
```

## License

MIT
