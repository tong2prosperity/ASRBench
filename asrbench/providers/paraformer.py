import asyncio
import os
import tempfile
import time

import dashscope
from dashscope.audio.asr import Recognition
from http import HTTPStatus

from .base import ASRProvider, ASRResult


class ParaformerProvider(ASRProvider):
    name = "paraformer"
    supported_formats = frozenset({"wav", "pcm"})

    def __init__(self, config: dict):
        self.model = config.get("model", "paraformer-realtime-v2")
        self.api_key = config.get("api_key", "") or os.environ.get("DASHSCOPE_API_KEY", "")

    async def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_recognize, audio_data, audio_format, sample_rate)

    def _sync_recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        ext = {"mp3": ".mp3", "wav": ".wav", "pcm": ".pcm"}.get(audio_format, ".wav")
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        try:
            t0 = time.monotonic()
            recognition = Recognition(
                model=self.model,
                format=audio_format,
                sample_rate=sample_rate,
                language_hints=["zh", "en"],
                callback=None,
            )
            result = recognition.call(tmp_path, api_key=self.api_key)
            elapsed = (time.monotonic() - t0) * 1000
            if result.status_code == HTTPStatus.OK:
                sentences = result.get_sentence()
                text = _extract_sentences(sentences)
                return ASRResult(text=text, duration_ms=elapsed, raw_response={"sentence": sentences})
            return ASRResult(text="", duration_ms=elapsed, raw_response={"error": result.message})
        finally:
            os.unlink(tmp_path)


def _extract_sentences(sentences) -> str:
    """Extract text from Recognition result, handling single dict or list."""
    if isinstance(sentences, dict):
        return sentences.get("text", "")
    if isinstance(sentences, list):
        return "".join(s.get("text", "") for s in sentences if isinstance(s, dict))
    return str(sentences) if sentences else ""
