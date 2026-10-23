"""Text-only end-of-turn classification over a Chat Completions endpoint."""

import json
from pathlib import Path
from urllib.parse import urlsplit

from .metrics import distribution

RESOURCE_DIR = Path(__file__).parent / "resources" / "eot"
METRIC_VERSION = "eot-binary-v1"
LABEL_POLICIES = ("original", "stop_is_complete")


def defaults():
    return {
        "prompt_template": (RESOURCE_DIR / "prompt.txt").read_text(encoding="utf-8"),
        "system_prompt": "",
        "label_policy": "original",
        "builtin_count": 1134,
        "builtin_name": "EOT 基础文本集（1134 条）",
    }


def validate_prompt(template):
    if not isinstance(template, str) or not template.strip():
        raise ValueError("EOT prompt 不能为空")
    if (template.count("{text}"), template.count("%s")) not in ((1, 0), (0, 1)):
        raise ValueError("EOT prompt 必须包含且仅包含一个 {text} 或 %s 占位符")
    return template


def validate_manifest(path):
    samples, errors, seen = [], [], set()
    with Path(path).expanduser().open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("每条清单必须是 JSON 对象")
                text, label = row.get("text"), row.get("label")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text 必须为非空文本")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("label 必须是整数 0 或 1")
                sid = row.get("sample_id", row.get("id", f"eot-{number:04d}"))
                if (
                    isinstance(sid, bool)
                    or not isinstance(sid, (str, int))
                    or not str(sid).strip()
                ):
                    raise ValueError("样本 ID 必须为非空字符串或整数")
                sid = str(sid).strip()
                if sid in seen:
                    raise ValueError("样本 ID 重复")
                seen.add(sid)
                category = row.get("category", "unknown")
                if not isinstance(category, str) or not category.strip():
                    raise ValueError("category 必须为非空字符串")
                samples.append(
                    {"id": sid, "text": text, "label": label, "category": category}
                )
            except (ValueError, TypeError) as exc:
                errors.append({"line": number, "error": str(exc)})
    if not samples and not errors:
        errors.append({"error": "数据集为空"})
    return {"samples": samples, "errors": errors, "warnings": [], "count": len(samples)}


def effective_label(sample, policy):
    return (
        1
        if policy == "stop_is_complete" and sample["category"] == "silence_request"
        else sample["label"]
    )


def parse_output(text):
    # Do not extract a random digit from an explanation or a contradictory response.
    if not isinstance(text, str):
        return None
    value = text.strip()
    return int(value) if value in ("0", "1") else None


def validate_params(params):
    base = params.get("base_url", "")
    if not isinstance(base, str):
        raise ValueError("EOT base_url 必须为文本地址")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("EOT base_url 必须是无凭据和查询参数的 HTTP(S) 地址")
    if not isinstance(params.get("model"), str) or not params["model"].strip():
        raise ValueError("EOT 必须配置 model")
    timeout = params.get("timeout", 30)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= 300
    ):
        raise ValueError("EOT timeout 必须在 0–300 秒之间")
    tokens = params.get("max_tokens", 4)
    if type(tokens) is not int or not 1 <= tokens <= 4096:
        raise ValueError("EOT max_tokens 必须为 1–4096 的整数")
    temp = params.get("temperature", 0)
    if (
        isinstance(temp, bool)
        or not isinstance(temp, (int, float))
        or not 0 <= temp <= 2
    ):
        raise ValueError("EOT temperature 必须在 0–2 之间")
    extra = params.get("extra_body", {})
    if not isinstance(extra, dict) or set(extra) & {
        "model",
        "messages",
        "stream",
        "max_tokens",
        "temperature",
        "api_key",
    }:
        raise ValueError("EOT extra_body 不能覆盖模型、消息、流式或基础采样参数")
    for key in ("prompt_template",):
        if key in params:
            validate_prompt(params[key])
    if "system_prompt" in params and not isinstance(params["system_prompt"], str):
        raise ValueError("EOT system_prompt 必须为文本")


async def call_model(client, params, text, settings):
    template = params.get("prompt_template", settings["prompt_template"])
    user = (
        template.replace("{text}", text)
        if "{text}" in template
        else template.replace("%s", text)
    )
    messages = []
    system = params.get("system_prompt", settings["system_prompt"])
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": params["model"],
        "messages": messages,
        "stream": False,
        "max_tokens": params.get("max_tokens", 4),
        "temperature": params.get("temperature", 0),
        **params.get("extra_body", {}),
    }
    base = params["base_url"].rstrip("/")
    endpoint = (
        base if base.endswith("/chat/completions") else base + "/chat/completions"
    )
    headers = (
        {"Authorization": "Bearer " + str(params["api_key"])}
        if params.get("api_key")
        else {}
    )
    response = await client.post(
        endpoint, headers=headers, json=payload, timeout=params.get("timeout", 30)
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    raw = choices[0].get("message", {}).get("content") if choices else None
    return {
        "raw_output": raw if isinstance(raw, str) else "",
        "pred": parse_output(raw),
    }


def classification_metrics(rows):
    valid = [r for r in rows if r["status"] == "success" and r.get("pred") in (0, 1)]
    tp = sum(r["label"] == 1 and r["pred"] == 1 for r in valid)
    fn = sum(r["label"] == 1 and r["pred"] == 0 for r in valid)
    fp = sum(r["label"] == 0 and r["pred"] == 1 for r in valid)
    tn = sum(r["label"] == 0 and r["pred"] == 0 for r in valid)

    def ratio(a, b):
        return a / b if b else None

    return {
        "count": len(valid),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": ratio(tp + tn, len(valid)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "fpr": ratio(fp, fp + tn),
    }


def summarize(run, rows):
    spec = run["spec"]
    groups = {
        p["profile_id"]: [r for r in rows if r["profile_id"] == p["profile_id"]]
        for p in spec["profiles"]
    }
    successful = [
        {r["sample_id"] for r in group if r["status"] == "success"}
        for group in groups.values()
    ]
    common = set.intersection(*successful) if successful else set()
    total = len(spec["samples"])
    output = []
    for p in spec["profiles"]:
        group = groups[p["profile_id"]]
        valid = [r for r in group if r["status"] == "success"]
        failed = sum(r["status"] in ("failed", "timeout") for r in group)
        invalid = sum(r["status"] == "invalid" for r in group)
        categories = {
            category: classification_metrics(
                [r for r in valid if r["category"] == category]
            )
            for category in sorted({s["category"] for s in spec["samples"]})
        }
        output.append(
            {
                "profile_id": p["profile_id"],
                "name": p["name"],
                "total": total,
                "completed": len(group),
                "success": len(valid),
                "request_failures": failed,
                "unparsed": invalid,
                "failures": failed + invalid,
                "failure_rate": (failed + invalid) / total if total else None,
                "cancelled": sum(r["status"] == "cancelled" for r in group),
                "metrics": classification_metrics(valid),
                "common_metrics": classification_metrics(
                    [r for r in valid if r["sample_id"] in common]
                ),
                "core_metrics": classification_metrics(
                    [r for r in valid if r["category"] != "silence_request"]
                ),
                "categories": categories,
                "request_elapsed_ms": distribution(
                    [
                        r["request_elapsed_ms"]
                        for r in group
                        if r["status"] in ("success", "invalid")
                    ]
                ),
                "valid_request_elapsed_ms": distribution(
                    [r["request_elapsed_ms"] for r in valid]
                ),
            }
        )
    return {
        "run_id": run["id"],
        "state": run["state"],
        "created": run["created"],
        "mode": "eot",
        "metric_version": spec["metric_version"],
        "common_count": len(common),
        "common_coverage": len(common) / total if total else None,
        "profiles": output,
        "spec": spec,
    }


def markdown_report(report):
    def val(v, pct=False):
        return "N/A" if v is None else f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    lines = [
        "# EOT 文本分类评测报告",
        "",
        f"任务：{report['run_id']}；标签策略：{report['spec']['eot']['label_policy']}",
        f"共同有效样本：{report['common_count']}；覆盖率：{val(report['common_coverage'], True)}",
        "",
        "|模型|有效/总数|准确率|精确率|召回率|误判率 FPR|共同准确率|请求失败|无法解析|P50 ms|P95 ms|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in report["profiles"]:
        m = p["metrics"]
        lat = p["request_elapsed_ms"]
        name = p["name"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            "|"
            + "|".join(
                [
                    name,
                    f"{p['success']}/{p['total']}",
                    *[
                        val(m[k], True)
                        for k in ("accuracy", "precision", "recall", "fpr")
                    ],
                    val(p["common_metrics"]["accuracy"], True),
                    str(p["request_failures"]),
                    str(p["unparsed"]),
                    val(lat["p50"]),
                    val(lat["p95"]),
                ]
            )
            + "|"
        )
    for p in report["profiles"]:
        name = p["name"].replace("|", "\\|").replace("\n", " ")
        lines.extend(
            [
                "",
                f"## {name} 分类明细",
                "",
                f"混淆矩阵：TP={p['metrics']['tp']}，FP={p['metrics']['fp']}，FN={p['metrics']['fn']}，TN={p['metrics']['tn']}。",
                "",
                "|类别|有效数|准确率|精确率|召回率|FPR|",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for category, metric in p["categories"].items():
            lines.append(
                "|"
                + "|".join(
                    [
                        category.replace("|", "\\|"),
                        str(metric["count"]),
                        *[
                            val(metric[k], True)
                            for k in ("accuracy", "precision", "recall", "fpr")
                        ],
                    ]
                )
                + "|"
            )
        lines += [
            "",
            f"排除 silence_request 的核心样本：{p['core_metrics']['count']}；准确率 {val(p['core_metrics']['accuracy'], True)}。",
        ]
    lines += [
        "",
        "1 表示轮次完成，0 表示未完成。模型必须严格返回 0 或 1；无法解析与请求失败分别统计，不算作有效分类。",
        "请求耗时包含完整 HTTP 请求与响应，不包含排队；不是音频语义结束点到断句信号的耗时。",
        "",
        "## 运行参数（脱敏）",
        "```json",
        json.dumps(report["spec"], ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines)
