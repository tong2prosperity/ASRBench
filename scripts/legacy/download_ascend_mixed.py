"""Download WTForbes/ASCEND_MIXED (test + validation) from Hugging Face
and convert to the corpus layout expected by ASRBench.

Output structure:
    corpus/ascend-mixed/
        test.tsv          (columns: path, sentence)
        dev.tsv           (validation split → dev.tsv for ASRBench compat)
        clips/
            <id>.wav      (one file per sample)

Usage:
    uv run python scripts/legacy/download_ascend_mixed.py
"""

import csv
import io
import os

from datasets import load_dataset

REPO_ID = "WTForbes/ASCEND_MIXED"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "ascend-mixed")
SPLITS = {"test": "test.tsv", "validation": "dev.tsv"}


def main():
    clips_dir = os.path.join(OUTPUT_DIR, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    for split_name, tsv_name in SPLITS.items():
        print(f"\n{'='*60}")
        print(f"Downloading split: {split_name}")
        print(f"{'='*60}")

        # Load without decoding audio — avoids torchcodec / torch dependency
        ds = load_dataset(REPO_ID, split=split_name)
        # Cast audio column to keep raw bytes
        ds = ds.cast_column("audio", ds.features["audio"])
        # Actually, let's just work with the raw arrow table
        table = ds.data  # pyarrow Table

        n = len(ds)
        print(f"  {n} samples loaded")

        tsv_path = os.path.join(OUTPUT_DIR, tsv_name)
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "sentence"], delimiter="\t")
            writer.writeheader()

            for i in range(n):
                row_id = table.column("id")[i].as_py()
                transcription = table.column("transcription")[i].as_py()

                # Audio column stores {"bytes": b"...", "path": "..."} in the arrow table
                audio_struct = table.column("audio")[i].as_py()
                audio_bytes = audio_struct["bytes"]

                filename = f"{split_name}_{row_id}.wav"
                wav_path = os.path.join(clips_dir, filename)

                if audio_bytes:
                    with open(wav_path, "wb") as wf:
                        wf.write(audio_bytes)
                else:
                    print(f"  WARNING: no audio bytes for sample {row_id}")
                    continue

                writer.writerow({
                    "path": filename,
                    "sentence": transcription,
                })

                if (i + 1) % 50 == 0:
                    print(f"  [{split_name}] {i+1}/{n} samples exported")

        print(f"  [{split_name}] Done — {n} samples → {tsv_path}")

    print(f"\nAll done. Corpus saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
