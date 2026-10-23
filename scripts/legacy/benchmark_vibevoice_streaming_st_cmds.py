"""Benchmark VibeVoice ASR Streaming on the fixed ST-CMDS 500 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.audio_utils import load_audio_use_ffmpeg
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


PUNCT_RE = re.compile(
    r"[，。！？、；：“”‘’（）《》【】「」『』〈〉〔〕·．…—\-–~～"
    r",.!?;:\"'\(\)\[\]\{\}<>\-\s]"
)
SPEAKER_RE = re.compile(r"(?:^|\n)\s*Speaker\s+\d+\s*:\s*", re.IGNORECASE)


def normalize(text: str) -> str:
    return PUNCT_RE.sub("", unicodedata.normalize("NFKC", text)).lower().strip()


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def write_outputs(rows: list[dict], output: Path, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "path", "status", "reference", "hypothesis", "raw_output",
        "edit_distance", "ref_chars", "cer", "audio_sec", "infer_ms",
        "rtf", "chunks", "init_ms", "first_compute_ms", "mean_chunk_compute_ms",
        "estimated_first_output_ms", "error",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    valid = [row for row in rows if row["status"] == "ok"]
    total_edits = sum(int(row["edit_distance"]) for row in valid)
    total_chars = sum(int(row["ref_chars"]) for row in valid)
    total_audio = sum(float(row["audio_sec"]) for row in valid)
    total_infer = sum(float(row["infer_ms"]) for row in valid) / 1000
    summary = {
        **metadata,
        "count_ok": len(valid),
        "count_errors": len(rows) - len(valid),
        "total_edits": total_edits,
        "total_ref_chars": total_chars,
        "corpus_cer": total_edits / total_chars if total_chars else 0,
        "macro_cer": sum(float(row["cer"]) for row in valid) / len(valid) if valid else 0,
        "exact_matches": sum(int(row["edit_distance"]) == 0 for row in valid),
        "total_audio_sec": total_audio,
        "total_infer_sec": total_infer,
        "rtf": total_infer / total_audio if total_audio else 0,
        "average_latency_ms": sum(float(row["infer_ms"]) for row in valid) / len(valid) if valid else 0,
        "average_first_compute_ms": sum(float(row["first_compute_ms"]) for row in valid) / len(valid) if valid else 0,
        "average_estimated_first_output_ms": sum(float(row["estimated_first_output_ms"]) for row in valid) / len(valid) if valid else 0,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    manifest = dataset / "manifest.tsv"
    output = Path(args.output).expanduser().resolve()
    with manifest.open(encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle, delimiter="\t"))

    with open(Path(args.model).expanduser() / "preprocessor_config.json") as handle:
        config = json.load(handle)
    sample_rate = config["target_sample_rate"]
    frame_samples = config["speech_tok_compress_ratio"]
    chunk_samples = config["chunk_frames"] * frame_samples
    lookahead_samples = config["lookahead_frames"] * frame_samples
    window_samples = chunk_samples + lookahead_samples

    processor = VibeVoiceASRProcessor.from_pretrained(args.model)
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    ).to(args.device).eval()

    rows = []
    if args.resume and output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = {row["path"] for row in rows if row["status"] == "ok"}

    metadata = {
        "model": str(Path(args.model).expanduser().resolve()),
        "manifest": str(manifest),
        "device": args.device,
        "dtype": "float32",
        "attention": "sdpa",
        "chunk_seconds": chunk_samples / sample_rate,
        "lookahead_seconds": lookahead_samples / sample_rate,
        "window_seconds": window_samples / sample_rate,
    }

    for index, sample in enumerate(samples, 1):
        if sample["path"] in completed:
            continue
        row = {"path": sample["path"], "reference": sample["reference"]}
        try:
            audio, _ = load_audio_use_ffmpeg(
                str(dataset / sample["path"]), resample=True, target_sr=sample_rate
            )
            duration = len(audio) / sample_rate
            cursor = 0
            texts = []
            chunk_times = []
            synchronize(args.device)
            infer_start = time.perf_counter()
            state = model.init_streaming_state(processor.tokenizer)
            synchronize(args.device)
            init_ms = (time.perf_counter() - infer_start) * 1000
            while cursor < len(audio):
                window = np.zeros(window_samples, dtype=np.float32)
                segment = audio[cursor : cursor + window_samples]
                window[: len(segment)] = segment
                synchronize(args.device)
                start = time.perf_counter()
                features = model.encode_speech(
                    torch.from_numpy(window).unsqueeze(0).to(args.device)
                )
                text, state = model.streaming_generate_step(
                    audio_features=features,
                    streaming_state=state,
                    tokenizer=processor.tokenizer,
                    max_new_tokens=256,
                    temperature=0.0,
                )
                synchronize(args.device)
                chunk_times.append((time.perf_counter() - start) * 1000)
                texts.append(text)
                cursor += chunk_samples
            synchronize(args.device)
            infer_ms = (time.perf_counter() - infer_start) * 1000
            raw_output = "".join(texts)
            hypothesis = SPEAKER_RE.sub("", raw_output).strip()
            ref = normalize(sample["reference"])
            hyp = normalize(hypothesis)
            edits = edit_distance(ref, hyp)
            row.update(
                status="ok",
                hypothesis=hypothesis,
                raw_output=raw_output.replace("\n", "\\n"),
                edit_distance=edits,
                ref_chars=len(ref),
                cer=edits / len(ref) if ref else 0,
                audio_sec=f"{duration:.6f}",
                infer_ms=f"{infer_ms:.3f}",
                rtf=f"{infer_ms / 1000 / duration:.8f}",
                chunks=len(chunk_times),
                init_ms=f"{init_ms:.3f}",
                first_compute_ms=f"{chunk_times[0]:.3f}",
                mean_chunk_compute_ms=f"{sum(chunk_times) / len(chunk_times):.3f}",
                estimated_first_output_ms=f"{(min(duration, window_samples / sample_rate) * 1000 + init_ms + chunk_times[0]):.3f}",
                error="",
            )
        except Exception as exc:
            row.update(
                status="error", hypothesis="", raw_output="", edit_distance=0,
                ref_chars=0, cer=0, audio_sec=0, infer_ms=0, rtf=0,
                chunks=0, init_ms=0, first_compute_ms=0, mean_chunk_compute_ms=0,
                estimated_first_output_ms=0, error=repr(exc),
            )
        rows.append(row)
        if index % 10 == 0 or index == len(samples):
            write_outputs(rows, output, metadata)
            print(f"{index}/{len(samples)} {sample['path']} {row['status']}", flush=True)


if __name__ == "__main__":
    main()
