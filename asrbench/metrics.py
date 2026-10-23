"""Versioned text rules shared by CLI, web, and exports."""

import unicodedata
import jieba
import jiwer

METRIC_VERSION = "nfkc-unicode-punct-jieba-v1"


def _normalized_text(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text).lower()
        if not unicodedata.category(c).startswith("P")
    ).strip()


def normalize_text(text: str) -> str:
    """Character normalization; retains the original helper's whitespace contract."""
    return "".join(_normalized_text(text).split())


def sequences(text):
    normalized = _normalized_text(text)
    chars = "".join(normalized.split())
    words = [w for w in jieba.lcut(normalized) if w.strip()]
    return " ".join(chars), " ".join(words)


def score_pair(reference, hypothesis):
    rc, rw = sequences(reference)
    hc, hw = sequences(hypothesis)

    def score(r, h):
        if not r:
            return {"rate": None, "S": 0, "D": 0, "I": len(h.split()), "N": 0}
        result = jiwer.process_words(r, h)
        return {
            "rate": result.wer,
            "S": result.substitutions,
            "D": result.deletions,
            "I": result.insertions,
            "N": len(r.split()),
        }

    c, w = score(rc, hc), score(rw, hw)
    return {"cer": c["rate"], "wer": w["rate"], "char_edits": c, "word_edits": w}


def compute_cer(reference, hypothesis):
    return score_pair(reference, hypothesis)["cer"]


def compute_wer(reference, hypothesis):
    return score_pair(reference, hypothesis)["wer"]


def compute_corpus_metrics(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("ref/hyp length mismatch")
    scores = [score_pair(r, h) for r, h in zip(references, hypotheses)]

    def aggregate(key):
        counts = {k: sum(s[key][k] for s in scores) for k in ("S", "D", "I", "N")}
        return counts, (
            sum(counts[k] for k in ("S", "D", "I")) / counts["N"]
            if counts["N"]
            else None
        )

    c, cer = aggregate("char_edits")
    w, wer = aggregate("word_edits")
    return {
        "cer": cer,
        "wer": wer,
        "count": len(scores),
        "char_edits": c,
        "word_edits": w,
    }


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q / 100
    lo = int(k)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (k - lo)


def distribution(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }
