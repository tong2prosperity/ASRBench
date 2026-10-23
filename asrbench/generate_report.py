"""Generate a markdown benchmark report from results/*.json.

Reads the per-provider result JSON files written by the runner, recomputes
corpus-level CER/WER via jiwer (so the report is authoritative and not
dependent on whatever aggregation the runner used), and writes
`benchmark_report.md` into the same directory.

Run via: `uv run python -m asrbench.generate_report [--results-dir results]`
"""

import argparse
import json
import os
import statistics
from datetime import datetime

from .metrics import compute_corpus_metrics


def load_provider_results(results_dir: str) -> dict[str, dict]:
    out = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        if fname.startswith("benchmark_"):
            continue
        path = os.path.join(results_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  skipping {fname}: {e}")
            continue
        pname = data.get("provider") or os.path.splitext(fname)[0]
        out[pname] = data
    return out


def summarize_provider(data: dict) -> dict:
    samples = data.get("samples", [])
    total = len(samples)
    errors = [s for s in samples if "error" in s]
    valid = [s for s in samples if "error" not in s]

    refs = [s["ground_truth"] for s in valid]
    hyps = [s["hypothesis"] for s in valid]
    corpus = compute_corpus_metrics(refs, hyps)

    # Legacy reports use 0 only for formatting an empty historical result;
    # new task reports always preserve N/A via summary.py.
    corpus = {
        k: (0.0 if k in ("cer", "wer") and v is None else v) for k, v in corpus.items()
    }

    # Latency / RTF
    rtfs = []
    latencies = []
    for s in valid:
        audio_dur = s.get("audio_duration_ms", 0) or 0
        dur = s.get("duration_ms", 0) or 0
        latencies.append(dur)
        if audio_dur > 0:
            rtfs.append(dur / audio_dur)

    # Per-sample CER for distribution stats
    cers = [s.get("cer", 0) for s in valid]
    perfect = sum(1 for c in cers if c == 0.0)

    # Top-10 worst samples (highest CER) — useful for error analysis
    worst = sorted(valid, key=lambda s: s.get("cer", 0), reverse=True)[:10]

    return {
        "total": total,
        "valid": len(valid),
        "errors": len(errors),
        "cer": corpus["cer"],
        "wer": corpus["wer"],
        "mean_cer": statistics.mean(cers) if cers else 0.0,
        "mean_wer": statistics.mean([s.get("wer", 0) for s in valid]) if valid else 0.0,
        "median_cer": statistics.median(cers) if cers else 0.0,
        "perfect_rate": perfect / len(valid) if valid else 0.0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 95) if latencies else 0.0,
        "avg_rtf": statistics.mean(rtfs) if rtfs else 0.0,
        "median_rtf": statistics.median(rtfs) if rtfs else 0.0,
        "worst": worst,
        "error_samples": errors[:5],  # first 5 errored samples for diagnosis
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def format_markdown(summaries: dict[str, dict]) -> str:
    lines = []
    lines.append("# ASR Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Dataset: Common Voice zh-CN (`corpus/zh-CN/test.tsv`)")
    lines.append("")

    lines.append("## Metric definitions")
    lines.append("")
    lines.append(
        "- **CER** — corpus-level Character Error Rate. Computed as "
        "`total_edits / total_reference_chars` via `jiwer.wer` over "
        "character sequences of the entire normalized corpus. "
        "This is the standard academic aggregate and weights long "
        "samples more than short ones."
    )
    lines.append(
        "- **WER** — corpus-level Word Error Rate. Identical to CER "
        "but using `jieba.lcut` word segmentation instead of "
        "character split."
    )
    lines.append(
        "- **Mean CER/WER** — arithmetic mean of per-sample scores. "
        "Included for comparison; not the standard metric."
    )
    lines.append(
        "- **Normalization** — NFKC + strip Chinese/ASCII "
        "punctuation + strip whitespace + lowercase. Applied to "
        "both reference and hypothesis before scoring."
    )
    lines.append(
        "- **RTF** — Real-Time Factor = "
        "`provider_elapsed_ms / audio_duration_ms`. Lower is faster."
    )
    lines.append("")

    # Main ranking table — sorted by corpus CER
    lines.append("## Results (sorted by CER)")
    lines.append("")
    lines.append(
        "| Rank | Provider | CER % | WER % | Mean CER % | Mean WER % | Avg RTF | P50 Latency | P95 Latency | Valid | Errors |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    ranked = sorted(summaries.items(), key=lambda kv: kv[1]["cer"])
    for rank, (pname, s) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | **{pname}** | {s['cer'] * 100:.2f} | {s['wer'] * 100:.2f} | "
            f"{s['mean_cer'] * 100:.2f} | {s['mean_wer'] * 100:.2f} | "
            f"{s['avg_rtf']:.3f} | {s['p50_latency_ms']:.0f}ms | "
            f"{s['p95_latency_ms']:.0f}ms | {s['valid']} | {s['errors']} |"
        )
    lines.append("")

    # Additional quality stats
    lines.append("## Quality distribution")
    lines.append("")
    lines.append("| Provider | Perfect (CER=0) % | Median CER | Median RTF |")
    lines.append("|---|---:|---:|---:|")
    for pname, s in ranked:
        lines.append(
            f"| {pname} | {s['perfect_rate'] * 100:.2f}% | "
            f"{s['median_cer'] * 100:.2f}% | {s['median_rtf']:.3f} |"
        )
    lines.append("")

    # Per-provider error analysis
    lines.append("## Per-provider error analysis")
    lines.append("")
    for pname, s in ranked:
        lines.append(f"### {pname}")
        lines.append("")
        if s["errors"]:
            lines.append(f"**{s['errors']} errored samples** (first 5 shown):")
            lines.append("")
            for e in s["error_samples"]:
                err = e.get("error", "?")
                lines.append(f"- `{e['id']}`: {err}")
            lines.append("")
        if s["worst"]:
            lines.append("**Top 10 highest-CER samples:**")
            lines.append("")
            lines.append("| CER | Ground Truth | Hypothesis |")
            lines.append("|---:|---|---|")
            for w in s["worst"]:
                gt = (w.get("ground_truth") or "").replace("|", "\\|")
                hyp = (w.get("hypothesis") or "").replace("|", "\\|")
                lines.append(f"| {w.get('cer', 0) * 100:.1f}% | {gt} | {hyp} |")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate ASRBench markdown report")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory with per-provider JSON files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (default: <results-dir>/benchmark_report.md)",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    output_path = args.output or os.path.join(results_dir, "benchmark_report.md")

    data_by_provider = load_provider_results(results_dir)
    if not data_by_provider:
        print(f"No result files found in {results_dir}")
        return

    summaries = {}
    for pname, data in data_by_provider.items():
        print(f"Summarizing {pname}...")
        summaries[pname] = summarize_provider(data)

    md = format_markdown(summaries)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport written to {output_path}")

    # Also dump a machine-readable summary alongside the markdown.
    json_path = os.path.join(results_dir, "benchmark_summary.json")
    compact = {
        pname: {k: v for k, v in s.items() if k not in ("worst", "error_samples")}
        for pname, s in summaries.items()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    print(f"Summary JSON written to {json_path}")


if __name__ == "__main__":
    main()
