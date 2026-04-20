import argparse
import asyncio
import json
import logging
import os
import sys
import time

import yaml
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .audio import load_audio, ensure_format
from .dataset import load_dataset
from .metrics import compute_cer, compute_wer, compute_corpus_metrics
from .report import save_results, print_summary
from .providers import PROVIDER_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_audio(sample_audio_path, supported_formats):
    """Synchronous audio loading + conversion, meant to run in executor."""
    audio_data, audio_fmt, duration_ms = load_audio(sample_audio_path)
    audio_data, audio_fmt = ensure_format(audio_data, audio_fmt, supported_formats, sample_rate=16000)
    return audio_data, audio_fmt, duration_ms


async def evaluate_provider(provider, samples, concurrency, output_dir, existing_results=None):
    """Evaluate a single provider on all samples with concurrency control."""
    sem = asyncio.Semaphore(concurrency)
    results = []
    errors = 0
    done = existing_results or {}
    loop = asyncio.get_event_loop()

    async def process_one(sample):
        nonlocal errors
        if sample.id in done:
            results.append(done[sample.id])
            return

        audio_data, audio_fmt, audio_duration_ms = await loop.run_in_executor(
            None, _prepare_audio, sample.audio_path, provider.supported_formats
        )

        async with sem:
            try:
                asr_result = await provider.recognize(audio_data, audio_fmt, 16000)
                asr_result.audio_duration_ms = audio_duration_ms
                cer = compute_cer(sample.ground_truth, asr_result.text)
                wer = compute_wer(sample.ground_truth, asr_result.text)
                entry = {
                    "id": sample.id,
                    "ground_truth": sample.ground_truth,
                    "hypothesis": asr_result.text,
                    "cer": cer,
                    "wer": wer,
                    "duration_ms": asr_result.duration_ms,
                    "audio_duration_ms": audio_duration_ms,
                }
                # If the provider returned empty text but didn't raise, treat it
                # as a soft failure so --resume will retry the sample. Common
                # causes: server-side rate limit, upstream rejection without
                # exception, truncated websocket responses.
                if not (asr_result.text or "").strip():
                    entry["error"] = "empty hypothesis"
                    # Keep a breadcrumb of what the provider saw so we can diagnose.
                    raw_err = None
                    if isinstance(asr_result.raw_response, dict):
                        raw_err = (
                            asr_result.raw_response.get("error")
                            or asr_result.raw_response.get("code")
                            or asr_result.raw_response.get("message")
                        )
                    if raw_err:
                        entry["error"] = f"empty hypothesis: {raw_err}"
                    errors += 1
                results.append(entry)
            except Exception as e:
                log.error(f"[{provider.name}] Error on {sample.id}: {e}")
                errors += 1
                results.append({
                    "id": sample.id,
                    "ground_truth": sample.ground_truth,
                    "hypothesis": "",
                    "cer": 1.0,
                    "wer": 1.0,
                    "duration_ms": 0,
                    "audio_duration_ms": audio_duration_ms,
                    "error": str(e),
                })

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"{provider.name}", total=len(samples))
        batch_size = concurrency * 2

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            await asyncio.gather(*[process_one(s) for s in batch])
            progress.advance(task, len(batch))
            # Checkpoint after every batch so a crash mid-run is resumable
            # via --resume without losing thousands of samples.
            save_results(
                {"provider": provider.name, "samples": results},
                output_dir,
                provider.name,
            )

    path = save_results({"provider": provider.name, "samples": results}, output_dir, provider.name)
    log.info(f"[{provider.name}] Results saved to {path}")

    valid = [r for r in results if "error" not in r]

    # Corpus-level CER/WER (weighted by reference length, standard aggregate).
    corpus = compute_corpus_metrics(
        [r["ground_truth"] for r in valid],
        [r["hypothesis"] for r in valid],
    )
    # Arithmetic mean of per-sample scores — kept for comparison / sanity checks.
    mean_cer = sum(r["cer"] for r in valid) / len(valid) if valid else 0
    mean_wer = sum(r["wer"] for r in valid) / len(valid) if valid else 0

    rtf_values = []
    for r in valid:
        audio_dur = r.get("audio_duration_ms", 0)
        if audio_dur > 0:
            rtf_values.append(r["duration_ms"] / audio_dur)
    avg_rtf = sum(rtf_values) / len(rtf_values) if rtf_values else 0

    log.info(
        f"[{provider.name}] corpus CER={corpus['cer']*100:.2f}% "
        f"WER={corpus['wer']*100:.2f}% "
        f"(mean CER={mean_cer*100:.2f}% WER={mean_wer*100:.2f}%) "
        f"RTF={avg_rtf:.3f} errors={errors}/{len(results)}"
    )

    return {
        "provider": provider.name,
        "cer": corpus["cer"],
        "wer": corpus["wer"],
        "mean_cer": mean_cer,
        "mean_wer": mean_wer,
        "avg_rtf": avg_rtf,
        "total": len(results),
        "errors": errors,
    }


def load_existing_results(output_dir: str, provider_name: str) -> dict:
    """Load existing results for resume support."""
    path = os.path.join(output_dir, f"{provider_name}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {s["id"]: s for s in data.get("samples", []) if "error" not in s}
    except Exception:
        return {}


async def main(args):
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    bench_cfg = config["bench"]
    project_root = os.path.dirname(config_path)
    corpus_dir = os.path.join(project_root, bench_cfg.get("corpus_dir", "corpus/zh-CN"))
    dataset_split = args.dataset or bench_cfg.get("dataset", "test")
    concurrency = args.concurrency or bench_cfg.get("concurrency", 10)
    output_dir = os.path.join(project_root, args.output or bench_cfg.get("output_dir", "results"))

    samples = load_dataset(corpus_dir, dataset_split)
    log.info(f"Loaded {len(samples)} samples from {dataset_split} split")

    if args.limit:
        samples = samples[:args.limit]
        log.info(f"Limited to {len(samples)} samples")

    provider_names = [args.provider] if args.provider else [
        name for name, pcfg in config["providers"].items() if pcfg.get("enabled", True)
    ]

    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    root_logger = logging.getLogger()
    for pname in provider_names:
        if pname not in PROVIDER_REGISTRY:
            log.warning(f"Unknown provider: {pname}, skipping")
            continue
        pcfg = config["providers"].get(pname, {})
        provider = PROVIDER_REGISTRY[pname](pcfg)

        # Per-provider log file — captures all INFO/ERROR for this run so we
        # have a durable record of errors, latency spikes, and summary lines
        # independent of the terminal scrollback.
        log_path = os.path.join(output_dir, f"{pname}.log")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        fh.setLevel(logging.INFO)
        root_logger.addHandler(fh)

        log.info(f"=== Starting evaluation: {provider.name} ({len(samples)} samples, concurrency={concurrency}) ===")

        existing = load_existing_results(output_dir, pname) if args.resume else {}
        if existing:
            log.info(f"  Resuming with {len(existing)} cached results")

        try:
            summary = await evaluate_provider(provider, samples, concurrency, output_dir, existing)
            summaries.append(summary)
        except Exception as e:
            log.error(f"[{pname}] Provider failed: {e}")
        finally:
            await provider.close()
            root_logger.removeHandler(fh)
            fh.close()

    if summaries:
        print_summary(summaries)


def cli():
    parser = argparse.ArgumentParser(description="ASRBench - ASR Evaluation Framework")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=["test", "dev", "both"], help="Dataset split")
    parser.add_argument("--provider", help="Run only this provider")
    parser.add_argument("--concurrency", type=int, help="Max concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
