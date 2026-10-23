"""Tencent 16k_zh_en vs 16k_zh_en_2.0 on meeting-farfield-500 (+ AliMeeting near-field twin).

Each model/condition was run twice so run-to-run variation of the streaming
service can be separated from the model difference. Empty hypotheses left after
the --resume retry are real outputs and are scored as full deletions (unlike
the runner's summary, which drops them).
"""
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import jiwer

from asrbench.metrics import normalize_text

ROOT = Path(__file__).resolve().parents[2] / "results/meeting-farfield-500"
CORPORA = {
    "far": Path.home() / "Resources/audio/meeting-farfield-500",
    "near": Path.home() / "Resources/audio/meeting-farfield-500-alimeeting-near",
}
MODELS = ["16k_zh_en", "16k_zh_en_2.0"]
RUNS = ["run1", "run2"]


def edits(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0, 0
    if not h:
        return len(r), len(r)
    m = jiwer.process_words(" ".join(r), " ".join(h))
    return m.substitutions + m.deletions + m.insertions, len(r)


def load(cond, model, run):
    data = json.load(open(ROOT / cond / model / run / "tencent_v2.json"))["samples"]
    out = {}
    for s in data:
        e, n = edits(s["ground_truth"], s["hypothesis"])
        out[s["id"]] = {"hyp": s["hypothesis"], "e": e, "n": n,
                        "service_error": ("error" in s and s["error"] != "empty hypothesis")}
    return out


def cer(ids, res):
    e = sum(res[i]["e"] for i in ids)
    n = sum(res[i]["n"] for i in ids)
    return 100 * e / n


def bootstrap(ids, a, b, iters=2000, seed=0):
    """95% CI of CER(a) - CER(b) in pp, a/b are id -> (edits, n) averaged over runs."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        smp = [ids[rng.randrange(len(ids))] for _ in ids]
        n = sum(a[i][1] for i in smp)
        diffs.append(100 * (sum(a[i][0] for i in smp) - sum(b[i][0] for i in smp)) / n)
    diffs.sort()
    return diffs[int(iters * 0.025)], diffs[int(iters * 0.975)]


def main():
    for cond, corpus in CORPORA.items():
        manifest = list(csv.DictReader(open(corpus / "manifest.tsv"), delimiter="\t"))
        groups = defaultdict(list)
        for row in manifest:
            groups["all"].append(row["path"])
            groups[row["corpus"]].append(row["path"])
            groups[f"{row['corpus']}/{row['stratum']}"].append(row["path"])
        res = {(m, r): load(cond, m, r) for m in MODELS for r in RUNS}
        svc = sum(v["service_error"] for d in res.values() for v in d.values())

        print(f"\n## {cond} ({len(manifest)} samples, residual service errors: {svc})\n")
        print("| subset | n | " + " | ".join(f"{m} CER % (run1 / run2)" for m in MODELS) + " | Δ (2.0 − old) pp | 95% CI |")
        print("|---|---:|" + "---:|" * len(MODELS) + "---:|---|")
        avg = {m: {i: (statistics.mean(res[(m, r)][i]["e"] for r in RUNS), res[(m, RUNS[0])][i]["n"])
                   for i in groups["all"]} for m in MODELS}
        for name, ids in groups.items():
            cells = []
            means = {}
            for m in MODELS:
                c = [cer(ids, res[(m, r)]) for r in RUNS]
                means[m] = statistics.mean(c)
                cells.append(f"**{means[m]:.2f}** ({c[0]:.2f} / {c[1]:.2f})")
            lo, hi = bootstrap(ids, avg[MODELS[1]], avg[MODELS[0]])
            print(f"| {name} | {len(ids)} | " + " | ".join(cells) +
                  f" | {means[MODELS[1]] - means[MODELS[0]]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")

        ids = groups["all"]
        same = lambda a, b: sum(res[a][i]["hyp"] == res[b][i]["hyp"] for i in ids)
        print(f"\nIdentical hypotheses: old run1==run2 {same((MODELS[0], 'run1'), (MODELS[0], 'run2'))}/{len(ids)}, "
              f"2.0 run1==run2 {same((MODELS[1], 'run1'), (MODELS[1], 'run2'))}/{len(ids)}, "
              f"old==2.0 (run1) {same((MODELS[0], 'run1'), (MODELS[1], 'run1'))}/{len(ids)}, "
              f"old==2.0 (run2) {same((MODELS[0], 'run2'), (MODELS[1], 'run2'))}/{len(ids)}")
        better = sum(avg[MODELS[1]][i][0] < avg[MODELS[0]][i][0] for i in ids)
        worse = sum(avg[MODELS[1]][i][0] > avg[MODELS[0]][i][0] for i in ids)
        print(f"Per-sample (run-averaged edits): 2.0 better {better}, worse {worse}, tie {len(ids) - better - worse}")

    # Near vs far on the same AliMeeting windows.
    print("\n## AliMeeting near vs far (same 250 windows, run-averaged)\n")
    near_ids = [r["path"] for r in csv.DictReader(open(CORPORA["near"] / "manifest.tsv"), delimiter="\t")]
    strata = {r["path"]: r["stratum"] for r in csv.DictReader(open(CORPORA["near"] / "manifest.tsv"), delimiter="\t")}
    print("| model | subset | near CER % | far CER % | far − near pp |")
    print("|---|---|---:|---:|---:|")
    for m in MODELS:
        for sub in ["all", "single", "overlap"]:
            ids = [i for i in near_ids if sub == "all" or strata[i] == sub]
            nc = statistics.mean(cer(ids, load("near", m, r)) for r in RUNS)
            fc = statistics.mean(cer(ids, load("far", m, r)) for r in RUNS)
            print(f"| {m} | {sub} | {nc:.2f} | {fc:.2f} | {fc - nc:+.2f} |")


if __name__ == "__main__":
    main()
