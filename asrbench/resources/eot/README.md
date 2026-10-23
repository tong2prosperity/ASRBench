# EOT 基础文本集

从旧 eot-bench 的 dataset.jsonl 原样迁入，共 1,134 条。只包含 text、label、category，没有历史评测输出或模型凭据。

label=1 表示轮次完成；label=0 表示未完成。保留原始 508 条正例、626 条负例，其中 200 条 silence_request 的原标签均为 0。

应用默认保留原标签。选择 stop_is_complete 时，仅在运行时把 silence_request 的有效标签改为 1；源文件和原标签保留，报告注明策略。核心指标始终另列排除停止类的 934 条。

|类别|完成 1|未完成 0|合计|
|---|---:|---:|---:|
|academic|157|124|281|
|conversation|72|49|121|
|customer_service|71|75|146|
|education|73|63|136|
|interview|69|68|137|
|professional|66|47|113|
|silence_request|0|200|200|

类别由旧项目从 TEN 的 finished、unfinished、wait 三组文本构建。prompt.txt 保留旧项目的默认用户 prompt；没有依赖缺失的 prompts.json。

这是文本二分类集，没有音频及语义结束点时间戳。可评估 EOT 分类和请求耗时，不能作为音频正确断句耗时的标注。模型应严格输出一个 0 或 1。
