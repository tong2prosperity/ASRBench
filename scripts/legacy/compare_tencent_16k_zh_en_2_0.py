"""Recompute old providers + tencent_v2 with the same metric code, plus paired v1 vs 2.0 stats."""
import json
import random
import statistics

from asrbench.metrics import compute_corpus_metrics, normalize_text
import jiwer

OLD = "/Users/trevorlink/Project/audio/ASRBenchOld/results"
NEW = "/Users/trevorlink/Project/audio/ASRBench/results/tencent_16k_zh_en_2.0"
SETS = {"zh-CN": (OLD, f"{NEW}/zh-CN"), "ascend-mixed": (f"{OLD}/ascend-mixed", f"{NEW}/ascend-mixed")}
PROVIDERS = ["qwen_asr", "volcengine", "funasr", "tencent", "paraformer"]


def load(path):
    return {s["id"]: s for s in json.load(open(path))["samples"]}


def summarize(samples):
    valid = [s for s in samples.values() if "error" not in s]
    corpus = compute_corpus_metrics([s["ground_truth"] for s in valid], [s["hypothesis"] for s in valid])
    rtf = [s["duration_ms"] / s["audio_duration_ms"] for s in valid if s["audio_duration_ms"] > 0]
    lat = sorted(s["duration_ms"] for s in valid)
    return {
        "cer": corpus["cer"] * 100, "wer": corpus["wer"] * 100,
        "perfect": 100 * sum(s["cer"] == 0 for s in valid) / len(valid),
        "rtf": statistics.mean(rtf), "p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95)],
        "valid": len(valid), "errors": len(samples) - len(valid),
    }


def edits(ref, hyp):
    r, h = list(normalize_text(ref)), list(normalize_text(hyp))
    if not r:
        return 0, 0
    if not h:
        return len(r), len(r)
    m = jiwer.process_words(" ".join(r), " ".join(h))
    return m.substitutions + m.deletions + m.insertions, len(r)


for name, (old_dir, new_dir) in SETS.items():
    print(f"\n## {name}")
    data = {p: load(f"{old_dir}/{p}.json") for p in PROVIDERS}
    data["tencent_v2 (16k_zh_en_2.0)"] = load(f"{new_dir}/tencent_v2.json")
    rows = sorted(((p, summarize(d)) for p, d in data.items()), key=lambda x: x[1]["cer"])
    print("| Provider | CER % | WER % | Perfect % | Avg RTF | P50 | P95 | Valid | Errors |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for p, s in rows:
        print(f"| {p} | {s['cer']:.2f} | {s['wer']:.2f} | {s['perfect']:.2f} | {s['rtf']:.3f} | {s['p50']:.0f}ms | {s['p95']:.0f}ms | {s['valid']} | {s['errors']} |")

    # Paired: samples valid in both v1 and 2.0
    v1, v2 = data["tencent"], data["tencent_v2 (16k_zh_en_2.0)"]
    ids = [i for i in v1 if i in v2 and "error" not in v1[i] and "error" not in v2[i]]
    pairs = [(edits(v1[i]["ground_truth"], v1[i]["hypothesis"]), edits(v2[i]["ground_truth"], v2[i]["hypothesis"])) for i in ids]
    e1 = sum(a[0] for a, _ in pairs); e2 = sum(b[0] for _, b in pairs); n = sum(a[1] for a, _ in pairs)
    better = sum(b[0] < a[0] for a, b in pairs); worse = sum(b[0] > a[0] for a, b in pairs)
    rng = random.Random(0); diffs = []
    for _ in range(2000):
        smp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        nn = sum(a[1] for a, _ in smp)
        diffs.append(100 * (sum(a[0] for a, _ in smp) - sum(b[0] for _, b in smp)) / nn)
    diffs.sort()
    print(f"\nPaired v1 vs 2.0 on {len(ids)} common valid samples: CER {100*e1/n:.2f}% -> {100*e2/n:.2f}%; "
          f"2.0 better {better}, worse {worse}, tie {len(ids)-better-worse}; "
          f"bootstrap 95% CI of (v1-2.0) = [{diffs[50]:+.2f}, {diffs[1949]:+.2f}] pp")
