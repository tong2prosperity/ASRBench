"""Run Tencent real-time ASR over the non-speech set and report false transcriptions.

Every clip is non-speech, so any non-empty hypothesis is a false positive.
Hypotheses are classified as:
  filler : only interjections like 嗯/啊/呃/哦 (after stripping punctuation)
  other  : any other text (hallucinated words)

Usage:
  python scripts/run_nonspeech_fp.py --config config_tencent_v2_zh.yaml \
      --variant 16k_zh_en:0 --variant 16k_zh_en_2.0:0 --variant 16k_zh_en_2.0:2
(variant = engine_model_type:filter_modal)
"""
import argparse
import asyncio
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from asrbench.audio import ensure_format, load_audio
from asrbench.metrics import normalize_text
from asrbench.providers.tencent_v2 import TencentV2Provider

FILLER = set("嗯啊呃哦噢唔哎欸诶呀哈嘿喔嗷额恩哼唉呵")


def classify(text):
    t = normalize_text(text)
    if not t:
        return "empty"
    return "filler" if all(ch in FILLER for ch in t) else "other"


async def run_variant(cfg, model, filter_modal, rows, corpus, out_dir, concurrency):
    provider = TencentV2Provider({**cfg, "engine_model_type": model, "filter_modal": filter_modal})
    sem = asyncio.Semaphore(concurrency)
    path = out_dir / f"{model}_fm{filter_modal}.json"
    done = {}
    if path.exists():
        done = {r["path"]: r for r in json.load(open(path)) if not r.get("error")}

    async def one(row):
        if row["path"] in done:
            return done[row["path"]]
        data, fmt, _ = load_audio(str(corpus / "clips" / row["path"]))
        data, fmt = ensure_format(data, fmt, provider.supported_formats, sample_rate=16000)
        async with sem:
            for attempt in range(3):
                r = await provider.recognize(data, fmt, 16000)
                err = r.raw_response.get("error")
                if not (err and ("4006" in err or "5000" in err or "Timeout" in err)):
                    break
                await asyncio.sleep(2 + attempt * 3)
        return {**row, "hypothesis": r.text, "error": err, "class": classify(r.text), "ms": round(r.duration_ms)}

    results = await asyncio.gather(*(one(r) for r in rows))
    json.dump(results, open(path, "w"), ensure_ascii=False, indent=1)
    return results


def report(all_results):
    variants = list(all_results)
    groups = defaultdict(list)
    for i, row in enumerate(all_results[variants[0]]):
        groups[row["group"]].append(i)
        groups[f"{row['group']}/{row['subgroup']}"].append(i)
    groups = dict(sorted(groups.items(), key=lambda kv: (kv[0].split("/")[0], "/" in kv[0], kv[0])))
    header = "| subset | n | " + " | ".join(f"{v} FP (filler / other)" for v in variants) + " |"
    print(header)
    print("|---|---:|" + "---|" * len(variants))
    total = list(range(len(all_results[variants[0]])))
    for name, idx in [("ALL", total)] + list(groups.items()):
        cells = []
        for v in variants:
            res = all_results[v]
            c = Counter(res[i]["class"] for i in idx)
            fp = c["filler"] + c["other"]
            cells.append(f"{fp} ({100 * fp / len(idx):.0f}%) = {c['filler']} / {c['other']}")
        bold = "**" if "/" not in name else ""
        print(f"| {bold}{name}{bold} | {len(idx)} | " + " | ".join(cells) + " |")
    for v in variants:
        errs = sum(1 for r in all_results[v] if r["error"])
        texts = Counter(normalize_text(r["hypothesis"]) for r in all_results[v] if r["class"] != "empty")
        print(f"\n{v}: service errors {errs}; most common outputs: {texts.most_common(12)}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_tencent_v2_zh.yaml")
    ap.add_argument("--corpus", type=Path, default=Path.home() / "Resources/audio/nonspeech-fp")
    ap.add_argument("--out", type=Path, default=Path("results/nonspeech-fp"))
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["providers"]["tencent_v2"]
    rows = list(csv.DictReader(open(args.corpus / "manifest.tsv"), delimiter="\t"))
    args.out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for v in args.variant:
        model, fm = v.rsplit(":", 1)
        all_results[f"{model} fm={fm}"] = await run_variant(cfg, model, int(fm), rows, args.corpus, args.out, args.concurrency)
        print(f"done {v}", flush=True)
    report(all_results)


if __name__ == "__main__":
    asyncio.run(main())
