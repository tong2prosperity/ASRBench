import json

import pytest

from asrbench.config import load_config, public_profiles, resolve_profile
from asrbench.dataset import validate_manifest
from asrbench.metrics import (
    compute_cer,
    compute_wer,
    compute_corpus_metrics,
    score_pair,
)


def test_manifest_mapping_normalization_and_ignored_hash(manifest):
    path, root = manifest
    result = validate_manifest(path, root)
    assert not result["errors"]
    assert result["count"] == 2
    assert result["samples"][0]["audio_duration_ms"] == 100
    assert "wav_sha256" not in str(result["samples"])


def test_missing_root_missing_audio_and_duplicate(manifest):
    path, root = manifest
    assert len(validate_manifest(path)["errors"]) == 2
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows.append(rows[0])
    path.write_text("\n".join(map(json.dumps, rows)))
    (root / "2.wav").unlink()
    result = validate_manifest(path, root)
    assert len(result["errors"]) == 2


def test_invalid_lines_empty_text_and_corrupt_audio(manifest):
    path, root = manifest
    path.write_text(
        "{bad\n"
        + json.dumps({"sample_id": "x", "audio_uri": "1.wav", "gold_text": ""})
        + "\n"
        + json.dumps({"sample_id": "z", "audio_uri": "2.wav", "gold_text": "中文"})
    )
    (root / "2.wav").write_bytes(b"not audio")
    result = validate_manifest(path, root)
    assert len(result["errors"]) == 3


def test_duration_warning_and_relative_path(manifest):
    path, root = manifest
    path.write_text(
        json.dumps(dict(sample_id="x", wav="1.wav", text="测试", duration_ms=2000))
    )
    result = validate_manifest(path, root)
    assert len(result["warnings"]) == 1
    assert result["samples"][0]["audio_duration_ms"] == 100


def test_empty_manifest(manifest):
    path, root = manifest
    path.write_text("")
    assert validate_manifest(path, root)["errors"]


def test_word_spaces_and_normalization():
    assert compute_wer("Hello WORLD！", "hello world") == 0
    assert compute_wer("hello world", "helloworld") > 0
    assert compute_cer("Ａ，你 好！", "a你好") == 0
    assert compute_cer("hello world", "helloworld") == 0


def test_empty_hypothesis_and_above_one():
    s = score_pair("你好", "")
    assert s["cer"] == 1
    assert s["char_edits"]["D"] == 2
    assert compute_cer("你", "你好吗啊") == 3


def test_corpus_weighted_and_all_failed():
    result = compute_corpus_metrics(["a", "abcdefghij"], ["", "abcdefghij"])
    assert result["cer"] == pytest.approx(1 / 11)
    assert result["char_edits"]["D"] == 1
    assert compute_corpus_metrics([], [])["cer"] is None


def test_config_legacy_env_redaction(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text(
        'providers:\n  funasr:\n    api_key: "${TEST_BENCH_SECRET}"\n    model: fake\n'
    )
    cfg = load_config(p)
    assert public_profiles(cfg)[0]["api_key"] == "<redacted>"
    with pytest.raises(ValueError, match="TEST_BENCH_SECRET"):
        resolve_profile(cfg["models"][0])
    monkeypatch.setenv("TEST_BENCH_SECRET", "test-only")
    assert resolve_profile(cfg["models"][0])["api_key"] == "test-only"


def test_duplicate_profile_rejected(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "models:\n- {profile_id: a, provider: funasr}\n- {profile_id: a, provider: funasr}\n"
    )
    with pytest.raises(ValueError, match="profile_id"):
        load_config(p)


def test_json_non_object_and_null_id(manifest):
    path, root = manifest
    path.write_text(
        "null\n[]\n" + json.dumps({"sample_id": None, "wav": "1.wav", "text": "你好"})
    )
    assert len(validate_manifest(path, root)["errors"]) == 3
