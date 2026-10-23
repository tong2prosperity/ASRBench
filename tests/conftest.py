import json
import wave

import pytest


@pytest.fixture
def config():
    return {
        "models": [
            dict(
                profile_id="alpha",
                name="模型 A",
                provider="funasr",
                enabled=True,
                params={"api_key": "test-only", "model": "fake-a"},
            ),
            dict(
                profile_id="beta",
                name="模型 B",
                provider="funasr",
                enabled=True,
                params={"api_key": "test-only", "model": "fake-b"},
            ),
        ]
    }


@pytest.fixture
def manifest(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    for i in (1, 2):
        with wave.open(str(audio / f"{i}.wav"), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(8000)
            out.writeframes(bytes(800 * i * 4))
    path = tmp_path / "data.jsonl"
    rows = [
        dict(
            sample_id=str(i),
            audio_uri=f"storage://customer/{i}.wav",
            gold_text="你好 world",
            duration_ms=100 * i,
            wav_sha256="ignored",
        )
        for i in (1, 2)
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return path, audio
