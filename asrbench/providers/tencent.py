"""Tencent Cloud ASR provider — streaming WebSocket via tencentcloud-speech-sdk-python.

Uses the streaming `SpeechRecognizer` (wss://asr.cloud.tencent.com/asr/v2) with
`16k_zh_en` engine — Chinese + English bilingual — which is **only** available
through the streaming API (the REST `SentenceRecognition` endpoint rejects it
with InvalidParameter).

The SDK is vendored at vendor/tencentcloud_speech_sdk/ because the package is
not published to PyPI. See CLAUDE.md for setup notes.
"""
import asyncio
import threading
import time

from .base import ASRProvider, ASRResult

# SDK imports use top-level `asr` / `common` packages (not namespaced) — see
# vendor/tencentcloud_speech_sdk/pyproject.toml for why we expose them this way.
from asr.speech_recognizer import SpeechRecognizer, SpeechRecognitionListener
from common.credential import Credential


# Tencent voice_format constants from the SDK docs.
# 1 = PCM, 12 = WAV. We send PCM (our ensure_format already normalizes).
_VOICE_FORMAT_PCM = 1
_VOICE_FORMAT_WAV = 12


class _Collector(SpeechRecognitionListener):
    """Thread-safe sink for the websocket callbacks.

    Callbacks fire on the websocket reader thread; the executor thread that
    owns this collector blocks on `done` until either recognition_complete
    or fail lands.
    """

    def __init__(self):
        self.sentences: list[str] = []
        self.error: str | None = None
        self.done = threading.Event()
        self.voice_id: str | None = None

    def on_recognition_start(self, response):
        self.voice_id = response.get("voice_id")

    def on_sentence_end(self, response):
        # Each `on_sentence_end` fires once per finalized sentence; we
        # concatenate them in order to rebuild the full transcript.
        result = response.get("result") or {}
        text = result.get("voice_text_str") or ""
        if text:
            self.sentences.append(text)

    def on_recognition_complete(self, response):
        self.done.set()

    def on_fail(self, response):
        self.error = response.get("message") or str(response)
        self.done.set()


class TencentProvider(ASRProvider):
    name = "tencent"
    # Streaming API wants 16-bit mono PCM; WAV also accepted. Skip mp3 so
    # ensure_format always down-samples Common Voice's 48k mp3s to 16k PCM.
    supported_formats = frozenset({"pcm", "wav"})

    def __init__(self, config: dict):
        self.app_id = str(config["app_id"])
        self.secret_id = config["secret_id"]
        self.secret_key = config["secret_key"]
        # 16k_zh_en (streaming only) is the Chinese + English bilingual model.
        self.engine_model_type = config.get("engine_model_type", "16k_zh_en")
        # 200ms chunks at 16k/16bit mono = 6400 bytes; the SDK example uses the
        # same value. Writing larger chunks doesn't help throughput on the
        # backend and risks back-pressure warnings from the SDK.
        self.slice_size = config.get("slice_size", 6400)

    async def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_recognize, audio_data, audio_format, sample_rate
        )

    def _sync_recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        t0 = time.monotonic()
        collector = _Collector()
        credential = Credential(self.secret_id, self.secret_key)
        recognizer = SpeechRecognizer(
            self.app_id, credential, self.engine_model_type, collector
        )
        # Keep punctuation ON so the metric behaviour matches the other
        # providers (normalize_text strips it before scoring anyway). We do
        # NOT enable num_mode conversion — the metric doesn't normalize
        # number scripts either, and switching per-provider would tilt the
        # comparison.
        recognizer.set_filter_dirty(0)
        recognizer.set_filter_modal(0)
        recognizer.set_filter_punc(0)
        recognizer.set_convert_num_mode(0)
        recognizer.set_word_info(0)
        recognizer.set_need_vad(1)
        recognizer.set_voice_format(
            _VOICE_FORMAT_PCM if audio_format == "pcm" else _VOICE_FORMAT_WAV
        )

        try:
            recognizer.start()
            # SDK's `write()` internally waits for status==OPENED before
            # sending, so we can stream chunks straight away.
            offset = 0
            while offset < len(audio_data):
                chunk = audio_data[offset:offset + self.slice_size]
                recognizer.write(chunk)
                offset += self.slice_size
            recognizer.stop()
            # stop() joins the websocket thread; after it returns, the
            # final recognition_complete callback has fired (or on_fail,
            # or the connection died). Wait briefly for the done signal
            # in case the callback hasn't set it yet (race between thread
            # join and last callback dispatch).
            collector.done.wait(timeout=10)
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return ASRResult(
                text="", duration_ms=elapsed, raw_response={"error": str(e)}
            )

        elapsed = (time.monotonic() - t0) * 1000
        if collector.error:
            return ASRResult(
                text="",
                duration_ms=elapsed,
                raw_response={"error": collector.error, "voice_id": collector.voice_id},
            )
        text = "".join(collector.sentences)
        return ASRResult(
            text=text,
            duration_ms=elapsed,
            raw_response={"voice_id": collector.voice_id, "sentences": len(collector.sentences)},
        )
