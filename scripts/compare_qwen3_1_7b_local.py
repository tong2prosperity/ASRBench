"""Build a combined ASCEND-MIXED report for local Qwen3-ASR 1.7B exports.

The report combines:
- Historical ASRBench JSON results under results/ascend-mixed.
- Historical local benchmark rows from results/ascend-mixed/local_mixed_cer_benchmark.txt.
- New local Qwen3-ASR 1.7B F16/Q6_K/Q4_K_M JSON results under results/ascend-mixed-local.

Run from the ASRBench repo root:
    uv run python scripts/compare_qwen3_1_7b_local.py
"""

from __future__ import annotations

import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from asrbench.metrics import compute_cer, compute_corpus_metrics, compute_wer, normalize_text


ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "results" / "ascend-mixed"
LOCAL_DIR = ROOT / "results" / "ascend-mixed-local"
OUT_DIR = ROOT / "results" / "ascend-mixed-qwen3-1.7b-comparison"


NEW_PROVIDERS = [
    "qwen3_asr_1_7b_f16",
    "qwen3_asr_1_7b_q6_k",
    "qwen3_asr_1_7b_q4_k_m",
]


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for sample in data.get("samples", []):
        if "error" not in sample:
            sample["cer"] = compute_cer(sample["ground_truth"], sample["hypothesis"])
            sample["wer"] = compute_wer(sample["ground_truth"], sample["hypothesis"])
    return data


def summarize_result(data: dict[str, Any]) -> dict[str, Any]:
    samples = data.get("samples", [])
    valid = [s for s in samples if "error" not in s]
    errors = [s for s in samples if "error" in s]
    corpus = compute_corpus_metrics(
        [s["ground_truth"] for s in valid],
        [s["hypothesis"] for s in valid],
    )
    rtfs = [
        s["duration_ms"] / s["audio_duration_ms"]
        for s in valid
        if s.get("audio_duration_ms", 0)
    ]
    latencies = [s.get("duration_ms", 0.0) for s in valid]
    cers = [s.get("cer", 0.0) for s in valid]
    wers = [s.get("wer", 0.0) for s in valid]
    total_edits = round(corpus["cer"] * sum(len(normalize_text(s["ground_truth"])) for s in valid))
    total_ref = sum(len(normalize_text(s["ground_truth"])) for s in valid)
    total_audio_s = sum(s.get("audio_duration_ms", 0) for s in valid) / 1000.0
    total_infer_s = sum(s.get("duration_ms", 0) for s in valid) / 1000.0
    return {
        "provider": data.get("provider"),
        "source": "json",
        "total": len(samples),
        "valid": len(valid),
        "errors": len(errors),
        "cer": corpus["cer"],
        "wer": corpus["wer"],
        "mean_cer": statistics.mean(cers) if cers else 0.0,
        "mean_wer": statistics.mean(wers) if wers else 0.0,
        "median_cer": statistics.median(cers) if cers else 0.0,
        "perfect_rate": sum(1 for c in cers if c == 0.0) / len(cers) if cers else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "avg_rtf": statistics.mean(rtfs) if rtfs else 0.0,
        "total_edits": total_edits,
        "total_ref": total_ref,
        "total_audio_s": total_audio_s,
        "total_infer_s": total_infer_s,
    }


def parse_old_local_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(FunASR-Q8_0|Qwen3-ASR-Q8_0|SenseVoice)\s+"
        r"([0-9.]+)%\s+(\d+)\s+(\d+)\s+([0-9.]+)x\s+([0-9.]+)s\s+([0-9.]+)ms",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        rows.append(
            {
                "provider": f"historical_local_{m.group(1).lower().replace('-', '_')}",
                "source": "local_mixed_cer_benchmark.txt",
                "total": int(m.group(3)) + int(m.group(4)),
                "valid": int(m.group(3)),
                "errors": int(m.group(4)),
                "cer": float(m.group(2)) / 100.0,
                "wer": None,
                "mean_cer": None,
                "mean_wer": None,
                "median_cer": None,
                "perfect_rate": None,
                "avg_latency_ms": float(m.group(7)),
                "p50_latency_ms": None,
                "avg_rtf": float(m.group(5)),
                "total_infer_s": float(m.group(6)),
            }
        )
    return rows


def normalized_edit_rate(a: str, b: str) -> float:
    # Character error rate of b when a is treated as reference.
    return compute_cer(a, b)


def compare_quantized(
    base: dict[str, Any],
    other: dict[str, Any],
) -> dict[str, Any]:
    base_samples = {s["id"]: s for s in base["samples"] if "error" not in s}
    other_samples = {s["id"]: s for s in other["samples"] if "error" not in s}
    ids = sorted(set(base_samples) & set(other_samples))
    changed = []
    raw_same = 0
    norm_same = 0
    better = 0
    worse = 0
    same_cer = 0
    edit_rates = []
    cer_deltas = []
    for sid in ids:
        b = base_samples[sid]
        o = other_samples[sid]
        if b["hypothesis"] == o["hypothesis"]:
            raw_same += 1
        if normalize_text(b["hypothesis"]) == normalize_text(o["hypothesis"]):
            norm_same += 1
        base_cer = compute_cer(b["ground_truth"], b["hypothesis"])
        other_cer = compute_cer(o["ground_truth"], o["hypothesis"])
        delta = other_cer - base_cer
        cer_deltas.append(delta)
        if delta < -1e-12:
            better += 1
        elif delta > 1e-12:
            worse += 1
        else:
            same_cer += 1
        edit_rate = normalized_edit_rate(b["hypothesis"], o["hypothesis"])
        edit_rates.append(edit_rate)
        if b["hypothesis"] != o["hypothesis"]:
            changed.append(
                {
                    "id": sid,
                    "ground_truth": b["ground_truth"],
                    "base_hypothesis": b["hypothesis"],
                    "other_hypothesis": o["hypothesis"],
                    "base_cer": base_cer,
                    "other_cer": other_cer,
                    "cer_delta": delta,
                    "hyp_edit_rate": edit_rate,
                }
            )
    changed.sort(key=lambda x: (abs(x["cer_delta"]), x["hyp_edit_rate"]), reverse=True)
    return {
        "provider": other["provider"],
        "baseline": base["provider"],
        "compared": len(ids),
        "raw_same": raw_same,
        "raw_same_rate": raw_same / len(ids) if ids else 0.0,
        "normalized_same": norm_same,
        "normalized_same_rate": norm_same / len(ids) if ids else 0.0,
        "better_than_baseline": better,
        "same_cer_as_baseline": same_cer,
        "worse_than_baseline": worse,
        "mean_cer_delta": statistics.mean(cer_deltas) if cer_deltas else 0.0,
        "mean_hyp_edit_rate": statistics.mean(edit_rates) if edit_rates else 0.0,
        "top_changed": changed[:20],
    }


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_report(summaries: list[dict[str, Any]], diffs: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(summaries, key=lambda s: (s["cer"] if s["cer"] is not None else 999))

    lines = [
        "# ASCEND-MIXED Qwen3-ASR 1.7B Export Benchmark",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Dataset: `corpus/ascend-mixed/test.tsv` (Chinese-English code-switching), 373 samples.",
        "",
        "Metrics: corpus CER/WER use ASRBench normalization. Local 1.7B runs use the same `encoder.int8.onnx` and differ only by LLM GGUF precision.",
        "",
        "## Combined Ranking",
        "",
        "| Rank | Provider | Source | CER % | WER % | Avg RTF | Avg Latency | Valid | Errors |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, s in enumerate(ranked, 1):
        avg_ms = "-" if s["avg_latency_ms"] is None else f"{s['avg_latency_ms']:.0f}ms"
        lines.append(
            f"| {i} | {s['provider']} | {s['source']} | {fmt_pct(s['cer'])} | "
            f"{fmt_pct(s['wer'])} | {fmt_float(s['avg_rtf'])} | {avg_ms} | "
            f"{s['valid']} | {s['errors']} |"
        )

    lines += [
        "",
        "## 1.7B Quantization Against F16",
        "",
        "| Quantized | Raw Same | Normalized Same | Better CER | Same CER | Worse CER | Mean CER Delta | Mean Hyp Edit Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in diffs:
        lines.append(
            f"| {d['provider']} | {d['raw_same']}/{d['compared']} ({d['raw_same_rate']*100:.2f}%) | "
            f"{d['normalized_same']}/{d['compared']} ({d['normalized_same_rate']*100:.2f}%) | "
            f"{d['better_than_baseline']} | {d['same_cer_as_baseline']} | {d['worse_than_baseline']} | "
            f"{d['mean_cer_delta']*100:+.3f}% | {d['mean_hyp_edit_rate']*100:.2f}% |"
        )

    lines += ["", "## Largest Quantization Deltas", ""]
    for d in diffs:
        lines += [
            f"### {d['provider']} vs {d['baseline']}",
            "",
            "| ID | Base CER % | Quant CER % | Delta % | Ref | F16 | Quant |",
            "|---|---:|---:|---:|---|---|---|",
        ]
        for row in d["top_changed"][:10]:
            lines.append(
                f"| {row['id']} | {row['base_cer']*100:.1f} | {row['other_cer']*100:.1f} | "
                f"{row['cer_delta']*100:+.1f} | {row['ground_truth']} | "
                f"{row['base_hypothesis']} | {row['other_hypothesis']} |"
            )
        lines.append("")

    (OUT_DIR / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "benchmark_summary.json").write_text(
        json.dumps({"summaries": summaries, "quantization_diffs": diffs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    results: list[dict[str, Any]] = []

    for path in sorted(HIST_DIR.glob("*.json")):
        if path.name.startswith("benchmark_"):
            continue
        results.append(load_result(path))

    local_results: dict[str, dict[str, Any]] = {}
    for provider in NEW_PROVIDERS:
        data = load_result(LOCAL_DIR / f"{provider}.json")
        local_results[provider] = data
        results.append(data)
        # Rewrite with per-sample WER filled in so ASRBench's stock report stays useful.
        (LOCAL_DIR / f"{provider}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summaries = [summarize_result(data) for data in results]
    summaries.extend(parse_old_local_rows(HIST_DIR / "local_mixed_cer_benchmark.txt"))

    base = local_results["qwen3_asr_1_7b_f16"]
    diffs = [
        compare_quantized(base, local_results["qwen3_asr_1_7b_q6_k"]),
        compare_quantized(base, local_results["qwen3_asr_1_7b_q4_k_m"]),
    ]

    write_report(summaries, diffs)
    print(f"Wrote {OUT_DIR / 'benchmark_report.md'}")
    print(f"Wrote {OUT_DIR / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
