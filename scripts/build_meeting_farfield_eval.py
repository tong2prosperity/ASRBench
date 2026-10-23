"""Build a ~500-utterance far-field / noisy meeting eval set from AliMeeting Eval + AISHELL-4 test.

Each sample is one annotated utterance of a target speaker, cut from channel 0 of
the 8-mic array (no beamforming), so it carries real room reverb, background
noise and, for the `overlap` stratum, other people talking at the same time.
The reference is the target speaker's transcript only.

Strata per corpus (seeded, round-robin across sessions so no meeting dominates):
  single  : no other speaker's annotated speech inside the padded window
  overlap : 20%-80% of the window overlapped by other speakers' speech

For AliMeeting the same windows are also cut from the target speaker's headset
recording into a sibling `*-near` corpus, so near-vs-far CER can be paired.

Output layout matches asrbench.dataset (test.tsv with path/sentence + clips/),
plus manifest.tsv with per-sample metadata.
"""
import argparse
import csv
import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

SR = 16000
PAD = 0.15  # seconds added on both sides of the annotated interval
MIN_DUR, MAX_DUR = 2.0, 20.0
MIN_CHARS = 6
OVERLAP_MIN, OVERLAP_MAX = 0.2, 0.8

# AISHELL-4 transcription marks. Unintelligible / noise / truncation marks make
# the reference unreliable, so such utterances are dropped; silence and filler
# wrappers are just stripped (same set the official recipe removes).
DROP_MARKS = ("<%>", "<$>", "<#>", "<->", "<_>")
STRIP_MARKS = ("<sil>", "<space>", "&", "`")


@dataclass
class Utt:
    corpus: str
    session: str
    speaker: str
    start: float
    end: float
    text: str
    far_wav: Path
    near_wav: Path | None = None
    overlap: float = 0.0
    stratum: str = ""
    meta: dict = field(default_factory=dict)


def parse_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    tiers: dict[str, list] = {}
    name = None
    xmin = xmax = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if m := re.match(r'name = "(.*)"', s):
            name = m.group(1)
            tiers[name] = []
        elif m := re.match(r"xmin = ([\d.]+)", s):
            xmin = float(m.group(1))
        elif m := re.match(r"xmax = ([\d.]+)", s):
            xmax = float(m.group(1))
        elif (m := re.match(r'text = "(.*)"\s*$', s)) and name is not None:
            text = m.group(1)
            if text.strip():
                tiers[name].append((xmin, xmax, text))
    return tiers


def clean_text(text: str) -> str | None:
    if any(mark in text for mark in DROP_MARKS):
        return None
    for mark in STRIP_MARKS:
        text = text.replace(mark, "")
    return text.strip()


def norm_len(text: str) -> int:
    t = unicodedata.normalize("NFKC", text)
    return sum(1 for ch in t if ch.isalnum())


def overlap_ratio(start, end, others):
    covered = []
    for s, e in others:
        lo, hi = max(start, s), min(end, e)
        if hi > lo:
            covered.append((lo, hi))
    covered.sort()
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in covered:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total / (end - start)


def session_utts(corpus, session, tiers, far_wav, near_for_speaker, duration):
    utts = []
    all_spans = {spk: [(s, e) for s, e, _ in ivs] for spk, ivs in tiers.items()}
    for spk, ivs in tiers.items():
        others = [span for o, spans in all_spans.items() if o != spk for span in spans]
        own = all_spans[spk]
        for s, e, raw in ivs:
            text = clean_text(raw)
            if not text or norm_len(text) < MIN_CHARS:
                continue
            ws, we = max(0.0, s - PAD), min(duration, e + PAD)
            if not (MIN_DUR <= we - ws <= MAX_DUR):
                continue
            # Skip if the padded window bleeds into the same speaker's
            # neighbouring utterance (its words would count as insertions).
            if any(o_s < we and o_e > ws for o_s, o_e in own if (o_s, o_e) != (s, e)):
                continue
            ratio = overlap_ratio(ws, we, others)
            stratum = "single" if ratio == 0 else ("overlap" if OVERLAP_MIN <= ratio <= OVERLAP_MAX else "")
            if not stratum:
                continue
            utts.append(Utt(corpus, session, spk, ws, we, text, far_wav, near_for_speaker(spk), ratio, stratum))
    return utts


def alimeeting(root: Path):
    far_dir = root / "Eval_Ali_far"
    near_dir = root / "Eval_Ali_near"
    utts = []
    for wav in sorted((far_dir / "audio_dir").glob("*.wav")):
        session = "_".join(wav.stem.split("_")[:2])  # R8001_M8004
        tiers = parse_textgrid(far_dir / "textgrid_dir" / f"{session}.TextGrid")
        info = sf.info(str(wav))
        assert info.samplerate == SR, wav

        def near(spk, session=session):
            p = near_dir / "audio_dir" / f"{session}_{spk}.wav"
            return p if p.exists() else None

        utts += session_utts("alimeeting", session, tiers, wav, near, info.duration)
    return utts


def aishell4(root: Path):
    utts = []
    for wav in sorted((root / "wav").glob("*.flac")):
        session = wav.stem
        tiers = parse_textgrid(root / "TextGrid" / f"{session}.TextGrid")
        info = sf.info(str(wav))
        assert info.samplerate == SR, wav
        utts += session_utts("aishell4", session, tiers, wav, lambda spk: None, info.duration)
    return utts


def pick(utts, n, rng):
    """Round-robin over sessions (shuffled) so every meeting contributes."""
    by_session = defaultdict(list)
    for u in utts:
        by_session[u.session].append(u)
    for lst in by_session.values():
        rng.shuffle(lst)
    sessions = sorted(by_session)
    rng.shuffle(sessions)
    chosen = []
    while len(chosen) < n and any(by_session.values()):
        for s in sessions:
            if by_session[s] and len(chosen) < n:
                chosen.append(by_session[s].pop())
    return chosen


def cut(src: Path, start: float, end: float, dst: Path, channel: int | None):
    data, sr = sf.read(str(src), start=int(start * SR), stop=int(end * SR), dtype="int16", always_2d=True)
    assert sr == SR
    mono = data[:, channel if channel is not None else 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), mono, SR, subtype="PCM_16")


def write_corpus(out: Path, rows):
    with open(out / "test.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "sentence"])
        for u in rows:
            w.writerow([u.meta["id"], u.text])
    with open(out / "manifest.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "corpus", "stratum", "overlap_ratio", "session", "speaker", "start", "end", "duration", "chars", "source", "sentence"])
        for u in rows:
            w.writerow([u.meta["id"], u.corpus, u.stratum, f"{u.overlap:.3f}", u.session, u.speaker,
                        f"{u.start:.3f}", f"{u.end:.3f}", f"{u.end - u.start:.3f}", norm_len(u.text),
                        u.meta["source"], u.text])
    digest = hashlib.sha256((out / "test.tsv").read_bytes()).hexdigest()
    (out / "test.tsv.sha256").write_text(digest + "\n")
    return digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alimeeting", type=Path, default=Path.home() / "Resources/audio/AliMeeting/Eval_Ali")
    ap.add_argument("--aishell4", type=Path, default=Path.home() / "Resources/audio/AISHELL-4/test")
    ap.add_argument("--out", type=Path, default=Path.home() / "Resources/audio/meeting-farfield-500")
    ap.add_argument("--per-corpus", type=int, default=250)
    ap.add_argument("--overlap-share", type=float, default=0.4)
    ap.add_argument("--seed", default="meeting-farfield-500-v1")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    candidates = {"alimeeting": alimeeting(args.alimeeting), "aishell4": aishell4(args.aishell4)}

    selected = []
    for corpus, utts in candidates.items():
        n_overlap = round(args.per_corpus * args.overlap_share)
        quotas = {"single": args.per_corpus - n_overlap, "overlap": n_overlap}
        for stratum, n in quotas.items():
            pool = [u for u in utts if u.stratum == stratum]
            got = pick(pool, n, rng)
            print(f"{corpus:10s} {stratum:8s} candidates={len(pool):5d} selected={len(got)}")
            selected += got

    selected.sort(key=lambda u: (u.corpus, u.session, u.start))
    far_out = args.out
    near_out = args.out.with_name(args.out.name + "-alimeeting-near")
    near_rows = []
    for i, u in enumerate(selected):
        uid = f"{u.corpus}_{u.stratum}_{u.session}_{u.speaker}_{int(u.start * 1000):07d}.wav"
        u.meta = {"id": uid, "source": f"{u.far_wav.name}:ch0"}
        cut(u.far_wav, u.start, u.end, far_out / "clips" / uid, channel=0)
        if u.near_wav is not None:
            nu = Utt(**{**u.__dict__, "meta": {"id": uid, "source": f"{u.near_wav.name}"}})
            cut(u.near_wav, u.start, u.end, near_out / "clips" / uid, channel=None)
            near_rows.append(nu)
        if (i + 1) % 100 == 0:
            print(f"  cut {i + 1}/{len(selected)}")

    print("far  :", far_out, len(selected), write_corpus(far_out, selected))
    if near_rows:
        print("near :", near_out, len(near_rows), write_corpus(near_out, near_rows))


if __name__ == "__main__":
    main()
