import asyncio
import base64
import os
import time
import threading

from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation,
    OmniRealtimeCallback,
    MultiModality,
)
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

from .base import ASRProvider, ASRResult

# Event type for final completed transcription
_COMPLETED_EVENT = "conversation.item.input_audio_transcription.completed"


class _QwenCallback(OmniRealtimeCallback):
    def __init__(self):
        self.final_transcripts: list[str] = []
        self.done_event = threading.Event()
        self.error = None
        self.raw = {}
        self.completed = False

    def on_event(self, response):
        self.raw = response
        if not isinstance(response, dict):
            return
        evt_type = response.get("type", "")
        if evt_type == _COMPLETED_EVENT:
            self.completed = True
            transcript = response.get("transcript", "")
            if transcript:
                self.final_transcripts.append(transcript)
            self.done_event.set()
        elif evt_type == "error":
            err = response.get("error") or response.get("message") or str(response)
            self.error = str(err)
            self.done_event.set()

    def on_open(self):
        pass

    def on_close(self, close_status_code, close_msg):
        if not self.completed and not self.error:
            self.error = "connection closed before completed transcription"
        self.done_event.set()


class QwenASRProvider(ASRProvider):
    name = "qwen_asr"
    supported_formats = frozenset({"pcm"})

    def __init__(self, config: dict):
        self.model = config.get("model", "qwen3-asr-flash-realtime")
        self.api_key = config.get("api_key", "") or os.environ.get(
            "DASHSCOPE_API_KEY", ""
        )

    async def recognize(
        self, audio_data: bytes, audio_format: str, sample_rate: int
    ) -> ASRResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_recognize, audio_data, audio_format, sample_rate
        )

    async def close(self):
        callback = getattr(self, "_callback", None)
        if callback:
            callback.done_event.set()
        conversation = getattr(self, "_conversation", None)
        if conversation:
            await asyncio.to_thread(conversation.close)

    def _sync_recognize(
        self, audio_data: bytes, audio_format: str, sample_rate: int
    ) -> ASRResult:
        t0 = time.monotonic()
        callback = _QwenCallback()

        conversation = OmniRealtimeConversation(
            model=self.model,
            callback=callback,
            api_key=self.api_key,
        )

        self._conversation = conversation
        self._callback = callback
        try:
            conversation.connect()
            conversation.update_session(
                output_modalities=[MultiModality.TEXT],
                enable_turn_detection=False,
                transcription_params=TranscriptionParams(
                    language="zh",
                    sample_rate=sample_rate,
                    input_audio_format="pcm",
                ),
            )

            chunk_size = sample_rate * 2 // 10  # 100ms chunks of 16-bit mono
            offset = 0
            while offset < len(audio_data):
                chunk = audio_data[offset : offset + chunk_size]
                encoded = base64.b64encode(chunk).decode("utf-8")
                conversation.append_audio(encoded)
                offset += chunk_size

            conversation.commit()
            if not callback.done_event.wait(timeout=30):
                raise TimeoutError("Qwen completed transcription timed out")
        finally:
            try:
                conversation.close()
            except Exception:
                pass

        elapsed = (time.monotonic() - t0) * 1000
        if callback.error:
            return ASRResult(
                text="", duration_ms=elapsed, raw_response={"error": callback.error}
            )
        text = "".join(callback.final_transcripts)
        return ASRResult(text=text, duration_ms=elapsed, raw_response=callback.raw)
