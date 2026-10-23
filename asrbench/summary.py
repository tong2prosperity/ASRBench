"""Authoritative reports used by every interface."""

import csv
import io
import json

from .metrics import METRIC_VERSION, compute_corpus_metrics, distribution


def summarize(run, rows):
    spec = run["spec"]
    profiles = spec["profiles"]
    total = len(spec["samples"])
    groups = {
        p["profile_id"]: [r for r in rows if r["profile_id"] == p["profile_id"]]
        for p in profiles
    }
    successes = {
        p: {r["sample_id"] for r in rs if r["status"] == "success"}
        for p, rs in groups.items()
    }
    common = set.intersection(*successes.values()) if successes else set()
    output = []
    for p in profiles:
        rs = groups[p["profile_id"]]
        valid = [r for r in rs if r["status"] == "success"]
        comparable = [r for r in valid if r["sample_id"] in common]

        def accuracy(items):
            return compute_corpus_metrics(
                [r["reference_text"] for r in items], [r["text"] for r in items]
            )

        elapsed = [r["request_elapsed_ms"] for r in valid]
        audio_total = sum(r["audio_duration_ms"] for r in valid)
        final = [
            r["final_latency_ms"]
            for r in valid
            if r.get("final_latency_ms") is not None
        ]
        output.append(
            {
                "profile_id": p["profile_id"],
                "name": p["name"],
                "total": total,
                "completed": len(rs),
                "success": len(valid),
                "failures": sum(r["status"] in ("failed", "timeout") for r in rs),
                "failure_rate": sum(r["status"] in ("failed", "timeout") for r in rs)
                / total
                if total
                else None,
                "cancelled": sum(r["status"] == "cancelled" for r in rs),
                "accuracy": accuracy(valid),
                "common_accuracy": accuracy(comparable),
                "request_elapsed_ms": distribution(elapsed),
                "rtf": sum(elapsed) / audio_total
                if audio_total and spec["mode"] == "batch"
                else None,
                "final_latency_ms": distribution(final),
                "final_coverage": len(final) / len(valid)
                if valid and spec["mode"] == "stream"
                else None,
                "first_text_latency_ms": distribution(
                    [
                        r["first_text_latency_ms"]
                        for r in valid
                        if r.get("first_text_latency_ms") is not None
                    ]
                ),
            }
        )
    return {
        "run_id": run["id"],
        "state": run["state"],
        "created": run["created"],
        "mode": spec["mode"],
        "metric_version": spec.get("metric_version", METRIC_VERSION),
        "common_count": len(common),
        "common_coverage": len(common) / total if total else None,
        "profiles": output,
        "spec": spec,
    }


def export_report(report, rows, format, events=None):
    if format == "json":
        return json.dumps(
            {"report": report, "attempts": rows, "events": events or []},
            ensure_ascii=False,
            indent=2,
        ), "application/json"
    if format == "csv":
        out = io.StringIO(newline="")
        fields = [
            "profile_id",
            "sample_id",
            "attempt_id",
            "status",
            "reference_text",
            "text",
            "cer",
            "wer",
            "request_elapsed_ms",
            "audio_duration_ms",
            "final_latency_ms",
            "final_reason",
            "error",
        ]
        if report["mode"] == "eot":
            fields = [
                "profile_id",
                "sample_id",
                "attempt_id",
                "status",
                "input_text",
                "category",
                "source_label",
                "label",
                "pred",
                "correct",
                "raw_output",
                "request_elapsed_ms",
                "error",
            ]
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = {
                k: (
                    "'" + v
                    if isinstance(v, str)
                    and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                    else v
                )
                for k, v in row.items()
            }
            writer.writerow(safe)
        return "\ufeff" + out.getvalue(), "text/csv; charset=utf-8"
    if format != "md":
        raise ValueError("导出格式必须为 json、csv 或 md")

    if report["mode"] == "eot":
        from .eot import markdown_report

        return markdown_report(report), "text/markdown; charset=utf-8"

    def val(v, pct=False):
        return "N/A" if v is None else f"{v * 100:.2f}%" if pct else f"{v:.2f}"

    lines = [
        "# ASR 评测报告",
        "",
        f"任务：{report['run_id']}；模式：{report['mode']}；状态：{report['state']}",
        f"计分版本：{report['metric_version']}；共同成功样本：{report['common_count']}，覆盖率：{val(report['common_coverage'], True)}",
        "",
        "|模型|成功/总数|CER|WER|共同 CER|共同 WER|请求 P50 ms|请求 P95 ms|RTF|final P50 ms|final P95 ms|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in report["profiles"]:
        name = p["name"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            "|"
            + "|".join(
                [
                    name,
                    f"{p['success']}/{p['total']}",
                    val(p["accuracy"]["cer"], True),
                    val(p["accuracy"]["wer"], True),
                    val(p["common_accuracy"]["cer"], True),
                    val(p["common_accuracy"]["wer"], True),
                    val(p["request_elapsed_ms"]["p50"]),
                    val(p["request_elapsed_ms"]["p95"]),
                    val(p["rtf"]),
                    val(p["final_latency_ms"]["p50"]),
                    val(p["final_latency_ms"]["p95"]),
                ]
            )
            + "|"
        )
    lines += [
        "",
        "CER/WER 为语料级编辑错误率；共同指标只使用所有模型均成功的样本。空识别文本按删除错误计分。",
        "批测 RTF 包含请求连接与上传，不包含排队和音频转换。流式请求总时长包含 10 秒静音观察窗口，不用于速度排名。",
        "ASR final 延迟相对音频末尾计算；不代表语义 EOT 或正确断句耗时。缺失值为 N/A，负延迟保留。",
        "",
        "## 运行参数（脱敏）",
        "```json",
        json.dumps(report["spec"], ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines), "text/markdown; charset=utf-8"
