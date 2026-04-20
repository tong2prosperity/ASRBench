import re
import unicodedata
import jiwer
import jieba


# Strip Chinese + ASCII punctuation after NFKC normalization. NFKC folds
# full-width ASCII (， ！ ？ etc.) to ASCII but leaves 。「」『』… as-is,
# so the regex has to cover both ranges.
_PUNCT_RE = re.compile(
    r"[，。！？、；：“”‘’（）《》【】「」『』〈〉〔〕·．…—\-–~～"
    r",.!?;:\"'\(\)\[\]\{\}<>\-\s]"
)


def normalize_text(text: str) -> str:
    """Normalize text for CER/WER: NFKC, strip punctuation + whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    return text.lower().strip()


def _char_seq(text: str) -> str:
    return " ".join(list(text))


def _word_seq(text: str) -> str:
    return " ".join(jieba.lcut(text))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Per-sample Character Error Rate."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(_char_seq(ref), _char_seq(hyp))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Per-sample Word Error Rate using jieba segmentation."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(_word_seq(ref), _word_seq(hyp))


def compute_corpus_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Corpus-level CER/WER: total edits / total reference tokens.

    This is the standard academic aggregate and differs from the arithmetic
    mean of per-sample scores — long samples weigh more than short ones, as
    they should. Empty references are skipped (jiwer errors on them).
    """
    assert len(references) == len(hypotheses), "ref/hyp length mismatch"
    refs_norm = [normalize_text(r) for r in references]
    hyps_norm = [normalize_text(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r]
    if not pairs:
        return {"cer": 0.0, "wer": 0.0, "count": 0}

    ref_chars = [_char_seq(r) for r, _ in pairs]
    hyp_chars = [_char_seq(h) for _, h in pairs]
    ref_words = [_word_seq(r) for r, _ in pairs]
    hyp_words = [_word_seq(h) for _, h in pairs]

    return {
        "cer": jiwer.wer(ref_chars, hyp_chars),
        "wer": jiwer.wer(ref_words, hyp_words),
        "count": len(pairs),
    }
