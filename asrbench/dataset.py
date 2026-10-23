import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse, unquote

from .audio import prepare_pcm


@dataclass
class Sample:
    id: str
    audio_path: str
    ground_truth: str
    audio_duration_ms: float = 0
    metadata: dict = field(default_factory=dict)


def load_dataset(corpus_dir: str, split: str) -> list[Sample]:
    splits = ["test", "dev"] if split == "both" else [split]
    samples = []
    for s in splits:
        with (Path(corpus_dir) / f"{s}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                samples.append(
                    Sample(
                        row["path"],
                        str(Path(corpus_dir) / "clips" / row["path"]),
                        row["sentence"],
                    )
                )
    return samples


def read_manifest(path, audio_root=None):
    path = Path(path).expanduser().resolve()
    root = Path(audio_root).expanduser().resolve() if audio_root else None
    samples, errors = [], []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = (
            list(csv.DictReader(f, delimiter="\t"))
            if path.suffix.lower() == ".tsv"
            else list(f)
        )
    for number, raw in enumerate(rows, 1):
        if isinstance(raw, str) and not raw.strip():
            continue
        try:
            row = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(row, dict):
                raise ValueError("每条清单必须为 JSON 对象")
            raw_id = row.get("sample_id", row.get("id", row.get("path", "")))
            if raw_id is None or isinstance(raw_id, (dict, list, bool)):
                raise ValueError("样本 ID 必须为非空字符串或数字")
            sid = str(raw_id).strip()
            if not sid or sid in seen:
                raise ValueError("样本 ID 缺失或重复")
            seen.add(sid)
            text = row.get("gold_text", row.get("text", row.get("sentence")))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("参考文本缺失或为空")
            uri = row.get(
                "audio_uri", row.get("wav", row.get("audio_path", row.get("path", "")))
            )
            if not isinstance(uri, str) or not uri:
                raise ValueError("音频路径缺失")
            parsed = urlparse(uri)
            if parsed.scheme == "storage":
                if not root:
                    raise ValueError("storage:// 音频必须指定 wav 根目录")
                audio = root / Path(unquote(parsed.path)).name
            elif parsed.scheme and parsed.scheme != "file":
                raise ValueError("仅支持本地路径、file:// 和 storage:// 文件名映射")
            else:
                audio = Path(
                    unquote(parsed.path) if parsed.scheme == "file" else uri
                ).expanduser()
                if not audio.is_absolute():
                    audio = (root if root else path.parent) / audio
            audio = audio.resolve()
            if not audio.is_file():
                raise ValueError(f"找不到音频：{audio}")
            samples.append(
                Sample(
                    sid,
                    str(audio),
                    text,
                    metadata={
                        "declared_duration_ms": row.get("duration_ms"),
                        "gold_end_ms": row.get("gold_end_ms"),
                    },
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"line": number, "error": str(exc)})
    return samples, errors


def validate_manifest(path, audio_root=None):
    samples, errors = read_manifest(path, audio_root)
    valid, warnings = [], []
    for sample in samples:
        try:
            pcm, duration = prepare_pcm(sample.audio_path)
            if not pcm or duration <= 0:
                raise ValueError("音频长度为零")
            sample.audio_duration_ms = duration
            stat = Path(sample.audio_path).stat()
            sample.metadata.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            declared = sample.metadata.get("declared_duration_ms")
            if declared is not None:
                try:
                    if abs(float(declared) - duration) > 20:
                        warnings.append(
                            {
                                "id": sample.id,
                                "warning": "清单时长与实际音频时长不同",
                                "actual_duration_ms": duration,
                            }
                        )
                except (ValueError, TypeError):
                    warnings.append(
                        {"id": sample.id, "warning": "清单时长无效，使用实际音频时长"}
                    )
            valid.append(asdict(sample))
        except Exception:
            errors.append({"id": sample.id, "error": "音频无法解码或长度无效"})
    if not samples and not errors:
        errors.append({"error": "数据集为空"})
    return {
        "samples": valid,
        "errors": errors,
        "warnings": warnings,
        "count": len(valid),
    }
