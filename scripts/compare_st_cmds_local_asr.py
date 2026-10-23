"""Compare the local ST-CMDS 500-result files with one metric implementation."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from benchmark_vibevoice_streaming_st_cmds import edit_distance, normalize


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = Path.home() / "Resources/audio/ST-CMDS-20170001_1-OS-test-500/manifest.tsv"


def load_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values), level))


def summarize(name: str, rows: list[dict]) -> dict:
    valid = [row for row in rows if not row.get("error") and row.get("status", "ok") == "ok"]
    edits = []
    ref_chars = []
    latencies = []
    audio_seconds = []
    for row in valid:
        ref = normalize(row["reference"])
        hyp = normalize(row["hypothesis"])
        edits.append(edit_distance(ref, hyp))
        ref_chars.append(len(ref))
        latencies.append(float(row["duration_ms"]))
        audio_seconds.append(float(row["audio_duration_ms"]) / 1000)
    total_audio = sum(audio_seconds)
    total_infer = sum(latencies) / 1000
    return {
        "model": name,
        "samples": len(rows),
        "valid": len(valid),
        "errors": len(rows) - len(valid),
        "empty_hypotheses": sum(not normalize(row["hypothesis"]) for row in valid),
        "total_edits": sum(edits),
        "total_ref_chars": sum(ref_chars),
        "corpus_cer": sum(edits) / sum(ref_chars),
        "macro_cer": statistics.mean(e / n for e, n in zip(edits, ref_chars)),
        "exact_matches": sum(edit == 0 for edit in edits),
        "total_audio_sec": total_audio,
        "total_infer_sec": total_infer,
        "rtf": total_infer / total_audio,
        "mean_latency_ms": statistics.mean(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "rows": valid,
        "edits": edits,
        "ref_chars": ref_chars,
    }


def duration_bands(summary: dict) -> list[dict]:
    bands = [
        ("<3 s", 0.0, 3.0),
        ("3–4 s", 3.0, 4.0),
        ("4–5 s", 4.0, 5.0),
        ("≥5 s", 5.0, float("inf")),
    ]
    output = []
    for label, low, high in bands:
        indexes = [
            index
            for index, seconds in enumerate(
                float(row["audio_duration_ms"]) / 1000 for row in summary["rows"]
            )
            if low <= seconds < high
        ]
        edits = sum(summary["edits"][index] for index in indexes)
        chars = sum(summary["ref_chars"][index] for index in indexes)
        output.append(
            {
                "model": summary["model"],
                "duration_band": label,
                "samples": len(indexes),
                "corpus_cer": edits / chars,
            }
        )
    return output


def paired_delta(left: dict, right: dict, seed: int = 20260904) -> dict:
    assert [row["id"] for row in left["rows"]] == [row["id"] for row in right["rows"]]
    left_edits = np.asarray(left["edits"], dtype=np.float64)
    right_edits = np.asarray(right["edits"], dtype=np.float64)
    chars = np.asarray(left["ref_chars"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(10_000, dtype=np.float64)
    for index in range(len(deltas)):
        sample = rng.integers(0, len(chars), size=len(chars))
        deltas[index] = (
            left_edits[sample].sum() / chars[sample].sum()
            - right_edits[sample].sum() / chars[sample].sum()
        )
    sample_delta = left_edits / chars - right_edits / chars
    return {
        "left": left["model"],
        "right": right["model"],
        "cer_delta": left["corpus_cer"] - right["corpus_cer"],
        "bootstrap_95_low": percentile(deltas.tolist(), 2.5),
        "bootstrap_95_high": percentile(deltas.tolist(), 97.5),
        "left_better": int(np.sum(sample_delta < 0)),
        "ties": int(np.sum(sample_delta == 0)),
        "left_worse": int(np.sum(sample_delta > 0)),
    }


def main() -> None:
    manifest = load_tsv(MANIFEST)
    funasr = load_tsv(RESULTS / "st_cmds_funasr_truncated.tsv")
    vibevoice = load_tsv(RESULTS / "st_cmds_vibevoice_streaming_1_5b_inclusive.tsv")
    qwen_raw = json.loads(
        (RESULTS / "st_cmds_qwen3_asr_1_7b_q6_k.json").read_text(encoding="utf-8")
    )["samples"]

    models = {
        "FunASR-Nano Q8": [
            {**row, "id": row["path"], "duration_ms": row["infer_ms"],
             "audio_duration_ms": float(row["audio_sec"]) * 1000}
            for row in funasr
        ],
        "Qwen3-ASR 1.7B Q6_K": [
            {**row, "id": row["id"], "reference": row["ground_truth"]}
            for row in qwen_raw
        ],
        "VibeVoice-ASR-Streaming 1.5B": [
            {**row, "id": row["path"], "duration_ms": row["infer_ms"],
             "audio_duration_ms": float(row["audio_sec"]) * 1000}
            for row in vibevoice
        ],
    }

    expected_ids = [row["path"] for row in manifest]
    expected_refs = {row["path"]: row["reference"] for row in manifest}
    checks = {}
    summaries = []
    for name, rows in models.items():
        checks[name] = {
            "row_count": len(rows),
            "unique_ids": len({row["id"] for row in rows}),
            "same_order_as_manifest": [row["id"] for row in rows] == expected_ids,
            "reference_mismatches": sum(
                row["reference"] != expected_refs[row["id"]] for row in rows
            ),
        }
        summaries.append(summarize(name, rows))

    comparisons = []
    for left in summaries:
        for right in summaries:
            if left["model"] < right["model"]:
                comparisons.append(paired_delta(left, right))

    serializable = []
    for summary in summaries:
        serializable.append({key: value for key, value in summary.items() if key not in {"rows", "edits", "ref_chars"}})

    vibe_rows = models["VibeVoice-ASR-Streaming 1.5B"]
    vibe_streaming = {
        "chunk_seconds": 2.933333333333333,
        "lookahead_seconds": 0.5333333333333333,
        "mean_init_ms": statistics.mean(float(row["init_ms"]) for row in vibe_rows),
        "mean_first_compute_ms": statistics.mean(float(row["first_compute_ms"]) for row in vibe_rows),
        "mean_estimated_first_output_ms": statistics.mean(float(row["estimated_first_output_ms"]) for row in vibe_rows),
        "p50_estimated_first_output_ms": statistics.median(float(row["estimated_first_output_ms"]) for row in vibe_rows),
        "p95_estimated_first_output_ms": percentile([float(row["estimated_first_output_ms"]) for row in vibe_rows], 95),
    }
    initial_vibe_path = RESULTS / "st_cmds_vibevoice_streaming_1_5b.tsv"
    vibe_repeatability = None
    if initial_vibe_path.exists():
        initial_rows = [
            {**row, "id": row["path"], "duration_ms": row["infer_ms"],
             "audio_duration_ms": float(row["audio_sec"]) * 1000}
            for row in load_tsv(initial_vibe_path)
        ]
        initial_summary = summarize("VibeVoice initial run", initial_rows)
        inclusive_summary = next(
            summary for summary in summaries
            if summary["model"] == "VibeVoice-ASR-Streaming 1.5B"
        )
        changed = sum(
            normalize(left["hypothesis"]) != normalize(right["hypothesis"])
            for left, right in zip(initial_rows, vibe_rows)
        )
        vibe_repeatability = {
            "initial_corpus_cer": initial_summary["corpus_cer"],
            "inclusive_corpus_cer": inclusive_summary["corpus_cer"],
            "cer_delta_percentage_points": (
                inclusive_summary["corpus_cer"] - initial_summary["corpus_cer"]
            ) * 100,
            "changed_normalized_hypotheses": changed,
            "samples": len(vibe_rows),
            "note": "The runs used the same inputs and decoding settings; MPS output varied slightly.",
        }
    output = {
        "dataset": {
            "manifest": str(MANIFEST),
            "samples": len(manifest),
            "sha256": "af390cb39619dd2f74600743ac59e346e918a49f354c51fa0023214d2683140b",
        },
        "quality_checks": checks,
        "summaries": serializable,
        "duration_bands": [row for summary in summaries for row in duration_bands(summary)],
        "paired_cer_comparisons": comparisons,
        "vibevoice_streaming": vibe_streaming,
        "vibevoice_repeatability": vibe_repeatability,
    }
    path = RESULTS / "st_cmds_three_model_comparison.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
