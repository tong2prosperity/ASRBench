# ASRBench

客户自有数据集上的多模型 ASR 对比工具。提供中文本地网页及命令行，评测 CER、WER、请求耗时和 RTF；腾讯适配器额外支持实时回放下的 ASR final 延迟。

一期不评估语义 EOT、断句正确率或“正确／错误断句耗时”。

## 安装和启动

需要 Python 3.10–3.12、[uv](https://docs.astral.sh/uv/) 和 FFmpeg。macOS 可执行 `brew install ffmpeg`，Ubuntu 可执行 `sudo apt install ffmpeg`。无需 Node。

```bash
uv sync --locked
cp config.yaml.example config.yaml
cp .env.example .env
# 编辑 .env，填写要比较的供应商凭据；按需修改 config.yaml 中的模型参数
uv run asrbench serve
```

打开 <http://127.0.0.1:8765>。服务只监听本机，关闭浏览器不停止后台任务，终端按 Ctrl+C 停止服务。

也可直接使用模板启动网页，配置凭据后再执行模型：

```bash
uv run asrbench serve --config config.yaml.example
```

依次完成：导入 JSONL 和 wav 根目录 → 多选模型 → 选择批测或实时回放 → 启动 → 查看报告、原音频和逐条差异 → 导出 CSV/JSON/Markdown。

音频会发送到所选模型的云端服务；清单、任务和评测结果保存在本机。默认不自动重试请求，避免重复调用。

## 客户数据集

JSONL 每行一条，例如以下人工示例：

```json
{"sample_id":"example-1","audio_uri":"storage://dataset/example-1.wav","gold_text":"你好，今天有什么安排？"}
```

- `sample_id`：唯一的非空样本 ID。
- `audio_uri`：本地路径、`file://` 路径，或 `storage://` 引用。
- `gold_text`：非空参考文本。也接受简单的 `wav`、`text` 字段。
- `storage://dataset/example-1.wav` 映射为所指定根目录下的 `example-1.wav`，不访问对象存储。
- 本地相对路径相对于清单所在目录解析；指定音频根目录后相对于该目录解析。
- 浏览器选择文件上传时，请填写音频根目录；大型清单可以直接填写服务器本机绝对路径。
- 兼容原 TSV 的 `path`、`sentence` 列；CLI 导入时将音频根目录指向 `clips/`。
- `duration_ms` 只辅助校验；计时采用实际音频时长。`gold_end_ms` 不参与一期评分；sha256 和其他来源信息不要求填写、不校验。

导入会检查重复 ID、空文本、规范化后无有效字符、缺失文件和解码错误。有错误的数据集不能启动任务；修正后重新导入。音频统一转换为 16kHz、单声道、16bit PCM。

客户提供的原始 JSONL、音频和现有评测数据不随代码分发。当前清单只有 `storage://` 引用，运行前仍需取得对应 wav。

## 模型配置

`config.yaml.example` 包含以下供应商配置：火山、FunASR、Qwen ASR、Paraformer、腾讯标准引擎、腾讯大模型 2.0。

每套配置有独立的 `profile_id`；复制一项并修改 `profile_id`、名称及模型参数，即可比较同一供应商的多个版本。`enabled: false` 可隐藏不使用的配置。

```yaml
models:
  - profile_id: my_tencent
    name: 腾讯方案 A
    provider: tencent_v2
    params:
      app_id: "${TENCENT_APP_ID}"
      secret_id: "${TENCENT_SECRET_ID}"
      secret_key: "${TENCENT_SECRET_KEY}"
      engine_model_type: 16k_zh_en_2.0
      filter_modal: 0
```

`.env` 从配置文件所在目录加载，不覆盖已设置的环境变量。仅执行所选模型时才解析其凭据。报告、网页和数据库配置快照会隐藏凭据值，模型请求的原始错误内容不直接写入报告。

旧版 `providers:` 配置仍可使用。阿里模型使用 `api_key`，腾讯使用 `app_id/secret_id/secret_key`，火山使用 `app_key/access_key`。模型是否开通、地址与资源权限需和客户账户匹配。

## 命令行

```bash
# 校验并导入；返回 dataset ID 和错误清单
uv run asrbench validate /path/to/eval.jsonl --audio-root /path/to/wavs

# 一次选择多个配置；先导入再运行
uv run asrbench run --manifest /path/to/eval.jsonl --audio-root /path/to/wavs \
  --models tencent_standard tencent_large_v2 --mode batch --concurrency 3

# 使用已导入的数据集
uv run asrbench run --dataset-id ds_XXXXXXXX --models funasr paraformer --limit 10

# 实时回放：并发固定 1，只支持腾讯两套适配器
uv run asrbench run --dataset-id ds_XXXXXXXX \
  --models tencent_standard tencent_large_v2 --mode stream

# 恢复未完成／取消的样本；不会重跑成功样本或自动重试失败请求
uv run asrbench resume run_XXXXXXXX

# 仅重试请求失败和超时的样本
uv run asrbench retry run_XXXXXXXX

# 导出；JSON/CSV 包含每次尝试，报告汇总使用每个模型每条样本的最新尝试
uv run asrbench report run_XXXXXXXX --format md --output exports/report.md
uv run asrbench report run_XXXXXXXX --format json --output exports/report.json
```

每个子命令均接受 `--config` 和 `--db`。网页和 CLI 默认共用 `runtime/asrbench.sqlite3`。同一个数据库只允许一个执行进程；网页已启动时请通过网页执行任务，或给独立 CLI 指定另一个数据库。只读报告导出可以与网页并行。

服务重启后，未完成任务标记为“已中断”，不会自动继续调用模型。恢复前检查音频文件大小和修改时间及模型参数；数据改变时需重新导入、新建任务。仅更新凭据值允许继续恢复。

旧命令 `python -m asrbench --provider ... --dataset ...` 仍支持批测，输出写入 `results/<run_id>/`；原 `--resume` 改为按 `run_id` 恢复。旧结果不会自动混入新任务。

## 指标口径

| 指标 | 定义 |
|---|---|
| CER | 字符级 `(替换＋删除＋插入)/参考字符数`，中文主指标 |
| WER | 中文 jieba 分词、英文保留词间边界后的词级错误率 |
| 请求耗时 | 调用模型开始到完整结果返回，包含连接和上传；不包含排队、解码、计分和写库 |
| RTF | 批测成功样本的总请求耗时／总音频时长 |
| ASR final 延迟 | 最后有效句段自然 final 接收时间，减去音频末尾基准时间 |
| 首文本延迟 | 流式模式首帧发送到首次非空识别文本的时间 |

文本计分统一执行 NFKC、转小写、移除 Unicode 标点。CER 忽略空白，WER 保留词间边界。不做数字读法转换、繁简转换、同义词替换或 LLM 纠错。

汇总使用语料级编辑次数之和，不取逐条错误率的简单平均。保留逐条分数和 S/D/I/N。空识别结果是一次有效识别，按删除错误计分；错误率可能超过 100%。全部失败或无分母时显示 N/A，不显示 0%。

各模型自身指标只统计成功样本，并列出失败率和覆盖数量。横向比较使用“共同 CER/WER”：只计所选模型全部成功的共同样本，必须一起检查共同覆盖率。失败率为失败与超时样本数／所选总样本数，取消和待执行样本另列。

### 腾讯实时回放

连接准备好后，以单调时钟每 200ms 发送对应 PCM，音频后继续发送完整 10 秒静音，再主动结束输入并清理会话。每条样本使用新连接。发送和接收独立执行。

音频末尾基准为“首帧发送时间＋实际音频时长”。final 延迟保留负值，并注明“音频结束前 final”；主动结束输入后返回的 final、没有有效句段 final、发送偏差超过 100ms 均不进入自然 final 延迟统计，保留缺失原因与事件时间线。

句段 final 与会话结束 `final=1` 分别记录；后者不能代替前者。延迟的均值、P50/P95 只针对有效样本，同时列出有效覆盖率。流式请求总时长包含观察窗口，不计算批测 RTF。

腾讯批测同样按其协议要求以 1:1 速度发送，随后主动结束输入；批测耗时不能当作自然 final 延迟。协议依据：[腾讯实时识别](https://cloud.tencent.com/document/product/1093/48982)、[腾讯 V2 实时识别](https://cloud.tencent.com/document/api/1093/131127)。

### EOT 边界

ASR final 只表示识别器确认了句段。语义 EOT 是独立模型判断用户是否说完，一期不调用该模型、不计算其准确率和断句耗时。

未来可把已有 EOT 文本评测接到独立任务类型，通过 Chat Completions 配置模型与 prompt；文本分类数据不能代替音频语义结束点标注。现有 ASR 原文、事件和时间信息可以作为后续分析输入。

## 数据和 API

- SQLite：`runtime/asrbench.sqlite3`；上传清单：`runtime/imports/`。
- 客户可自行保留音频目录；导入不搬移音频。
- `GET /api/models`、`POST/GET /api/datasets`：配置查询、数据校验与样本预览。
- `POST/GET /api/runs`：创建任务和任务列表。
- `POST /api/runs/{id}/stop|resume|retry`：生命周期操作。
- `GET /api/runs/{id}/results?offset=0&limit=50&history=false`：分页结果与文本差异。
- `GET /api/runs/{id}/report`、`export/json|csv|md`：同一计分源的报告与导出。
- `GET /api/runs/{id}/events?after=0`：持久化 SSE 事件，支持 `Last-Event-ID` 恢复订阅。
- 本机启动后可访问 `/docs` 查看接口参数。

CSV 为带 BOM 的 UTF-8，方便 Excel 打开；对可能作为公式解释的文本加前缀保护。JSON 保留原文。

## 验证与故障处理

```bash
uv sync --locked
uv run pytest -q
uv build
```

测试使用人工文本、临时生成的音频、模拟供应商与时钟，不需要密钥、不访问付费 ASR。

| 问题 | 处理 |
|---|---|
| storage 引用找不到音频 | 指定正确 wav 根目录，核对文件名；工具不会自动下载原始 FLV |
| 缺少环境变量 | 在配置文件同目录的 `.env` 填写所选供应商凭据 |
| 识别失败／超时 | 核对模型开通、凭据、配额和网络；修复后点击“重试失败样本” |
| 数据库由其他进程执行 | 使用现有网页，或关闭执行进程后再用 CLI；独立任务可指定其他 `--db` |
| final 指标缺失 | 查看该条 final 原因及事件时间线，不用请求耗时替代 |
| 任务停止 | 后续样本不会继续提交；已经发出的云端请求可能完成。SDK 执行中的同步调用可能稍后才释放 |

原实验脚本保留在 `scripts/legacy/`。其语料、历史结果和本机绝对路径不会被客户主流程使用。需要运行下载实验时使用 `uv sync --extra experiments`；这些脚本并非客户交付入口。

## Git 与隐私

仓库只提交代码、脱敏配置模板、人工测试夹具和文档。`.env`、私人配置、根目录真实 JSONL、客户需求原文、音频数据目录、结果、数据库、导出和日志已加入 `.gitignore`。测试数据均在临时目录生成。

不要把客户数据复制到源码或测试目录；新增真实数据请放 `data/` 或仓库外部目录。忽略规则不会自动清除已提交的文件，新增提交前仍需检查暂存文件列表。
