"""Build a non-speech false-positive eval set: every clip should transcribe to nothing.

Sources (all written as 16 kHz mono PCM16 wav):
  esc50/<class>      ESC-50 single-event clips (door, keyboard, footsteps, cough ...)
  meeting_far/<src>  3-8 s windows of channel 0 of the AliMeeting / AISHELL-4 arrays
                     where no speaker is annotated within +-0.5 s (room tone,
                     typing, chairs, doors — whatever the room had)
  meeting_near       same, from AliMeeting headset mics (breath, mic rubbing)
  synthetic/<kind>   digital silence, white / pink noise at several levels, 50 Hz hum

Output: <out>/clips/*.wav + manifest.tsv (path, group, subgroup, duration, source).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_meeting_farfield_eval import parse_textgrid  # noqa: E402

SR = 16000
MARGIN = 0.5

ESC_CLASSES = [
    "door_wood_knock", "door_wood_creaks", "keyboard_typing", "mouse_click",
    "footsteps", "clock_tick", "can_opening", "drinking_sipping", "pouring_water",
    "glass_breaking", "clapping", "breathing", "coughing", "sneezing", "snoring",
    "laughing", "vacuum_cleaner", "washing_machine", "water_drops", "wind",
]


def resample_to_16k(x, sr):
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr == SR:
        return x
    n = int(round(len(x) * SR / sr))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def write(out, name, x):
    x = np.clip(x, -1, 1)
    sf.write(str(out / "clips" / name), (x * 32767).astype(np.int16), SR, subtype="PCM_16")
    return len(x) / SR


def speech_free_windows(tiers, duration, rng, n, min_len=3.0, max_len=8.0):
    spans = sorted((s - MARGIN, e + MARGIN) for ivs in tiers.values() for s, e, _ in ivs)
    gaps, cur = [], 0.0
    for s, e in spans:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if duration > cur:
        gaps.append((cur, duration))
    gaps = [(s, e) for s, e in gaps if e - s >= min_len and s > 1.0]
    rng.shuffle(gaps)
    wins = []
    for s, e in gaps[:n]:
        length = rng.uniform(min_len, min(max_len, e - s))
        start = rng.uniform(s, e - length)
        wins.append((start, start + length))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", type=Path, default=Path.home() / "Resources/audio/ESC-50-master")
    ap.add_argument("--alimeeting", type=Path, default=Path.home() / "Resources/audio/AliMeeting/Eval_Ali")
    ap.add_argument("--aishell4", type=Path, default=Path.home() / "Resources/audio/AISHELL-4/test")
    ap.add_argument("--out", type=Path, default=Path.home() / "Resources/audio/nonspeech-fp")
    ap.add_argument("--esc-per-class", type=int, default=6)
    ap.add_argument("--seed", default="nonspeech-fp-v1")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    (args.out / "clips").mkdir(parents=True, exist_ok=True)
    rows = []

    # ESC-50
    meta = list(csv.DictReader(open(args.esc50 / "meta/esc50.csv")))
    for cls in ESC_CLASSES:
        files = sorted(r["filename"] for r in meta if r["category"] == cls)
        for fn in rng.sample(files, args.esc_per_class):
            x, sr = sf.read(str(args.esc50 / "audio" / fn))
            name = f"esc50_{cls}_{Path(fn).stem}.wav"
            dur = write(args.out, name, resample_to_16k(x, sr))
            rows.append([name, "esc50", cls, f"{dur:.2f}", fn])

    # Meeting rooms: far-field channel 0 (AliMeeting + AISHELL-4) and AliMeeting headsets
    sessions = []
    for wav in sorted((args.alimeeting / "Eval_Ali_far/audio_dir").glob("*.wav")):
        sess = "_".join(wav.stem.split("_")[:2])
        sessions.append(("alimeeting", sess, wav, args.alimeeting / "Eval_Ali_far/textgrid_dir" / f"{sess}.TextGrid"))
    for wav in sorted((args.aishell4 / "wav").glob("*.flac")):
        sessions.append(("aishell4", wav.stem, wav, args.aishell4 / "TextGrid" / f"{wav.stem}.TextGrid"))
    for corpus, sess, wav, tg in sessions:
        tiers = parse_textgrid(tg)
        per = 4 if corpus == "alimeeting" else 2  # 8 vs 20 sessions -> ~32 / ~40 clips
        for s, e in speech_free_windows(tiers, sf.info(str(wav)).duration, rng, per):
            x, _ = sf.read(str(wav), start=int(s * SR), stop=int(e * SR), always_2d=True)
            name = f"meetfar_{corpus}_{sess}_{int(s * 1000):07d}.wav"
            dur = write(args.out, name, x[:, 0])
            rows.append([name, "meeting_far", corpus, f"{dur:.2f}", f"{wav.name}:ch0@{s:.2f}-{e:.2f}"])
        if corpus == "alimeeting":
            nears = sorted((args.alimeeting / "Eval_Ali_near/audio_dir").glob(f"{sess}_*.wav"))
            for s, e in speech_free_windows(tiers, sf.info(str(nears[0])).duration, rng, 3):
                near = rng.choice(nears)
                x, _ = sf.read(str(near), start=int(s * SR), stop=int(e * SR))
                name = f"meetnear_{near.stem}_{int(s * 1000):07d}.wav"
                dur = write(args.out, name, x)
                rows.append([name, "meeting_near", "alimeeting", f"{dur:.2f}", f"{near.name}@{s:.2f}-{e:.2f}"])

    # Synthetic
    nrng = np.random.default_rng(12345)
    n = 5 * SR

    def db(level):
        return 10 ** (level / 20)

    def pink(size):
        white = nrng.standard_normal(size)
        spec = np.fft.rfft(white)
        spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
        p = np.fft.irfft(spec, size)
        return p / np.sqrt(np.mean(p ** 2))

    synth = [("silence", "digital", np.zeros(n))]
    for lvl in (-60, -45, -30, -20):
        for k in range(2):
            synth.append(("white_noise", f"{lvl}dBFS_{k}", nrng.standard_normal(n) * db(lvl)))
            synth.append(("pink_noise", f"{lvl}dBFS_{k}", pink(n) * db(lvl)))
    t = np.arange(n) / SR
    for lvl in (-40, -25):
        hum = sum(np.sin(2 * np.pi * 50 * h * t) / h for h in (1, 2, 3, 5))
        synth.append(("hum_50hz", f"{lvl}dBFS", hum / np.sqrt(np.mean(hum ** 2)) * db(lvl)))
    for kind, variant, x in synth:
        name = f"synthetic_{kind}_{variant}.wav"
        dur = write(args.out, name, x)
        rows.append([name, "synthetic", kind, f"{dur:.2f}", variant])

    with open(args.out / "manifest.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["path", "group", "subgroup", "duration", "source"])
        w.writerows(rows)
    from collections import Counter
    print(len(rows), Counter(r[1] for r in rows))


if __name__ == "__main__":
    main()
