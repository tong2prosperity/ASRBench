import csv
import os
from dataclasses import dataclass


@dataclass
class Sample:
    id: str
    audio_path: str
    ground_truth: str


def load_dataset(corpus_dir: str, split: str) -> list[Sample]:
    """Load samples from a TSV split (test / dev / both)."""
    splits = ["test", "dev"] if split == "both" else [split]
    samples = []
    for s in splits:
        tsv_path = os.path.join(corpus_dir, f"{s}.tsv")
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_path = os.path.join(corpus_dir, "clips", row["path"])
                samples.append(Sample(
                    id=row["path"],
                    audio_path=audio_path,
                    ground_truth=row["sentence"],
                ))
    return samples
