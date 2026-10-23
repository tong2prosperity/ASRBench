"""Tencent Cloud real-time ASR V2 (WebSocket) — `16k_zh_en_2.0` engine.

Docs: https://cloud.tencent.com/document/api/1093/131127

Same host/path and HMAC-SHA1 signing as the v1 streaming API
(`wss://asr.cloud.tencent.com/asr/v2/<appid>`), but the response schema is
different: results arrive as `sentences: {sentence_id, sentence_type, text,
...}` instead of `result.slice_type / voice_text_str`, so the vendored
SpeechRecognizer can't parse it. This provider talks to the socket directly.

Sending pattern mirrors TencentProvider: 16k mono PCM in 6400-byte (200ms)
chunks sent at 1:1 speed, then `{"type": "end"}`, then wait for final=1.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import random
import time
import urllib.parse
import uuid

import websockets

from .base import ASRProvider, ASRResult

_HOST_PATH = "asr.cloud.tencent.com/asr/v2/"
_VOICE_FORMAT_PCM = 1


class TencentV2Provider(ASRProvider):
    name = "tencent_v2"
    supported_formats = frozenset({"pcm"})

    def __init__(self, config: dict):
        self.options = config
        self.app_id = str(config["app_id"])
        self.secret_id = config["secret_id"]
        self.secret_key = config["secret_key"]
        self.engine_model_type = config.get("engine_model_type", "16k_zh_en_2.0")
        self.timeout = config.get("timeout", 60)
        # 0 keeps filler words (嗯/啊) so outputs are comparable with the v1 run;
        # 1 = partial, 2 = strict filtering.
        self.filter_modal = config.get("filter_modal", 0)

    def _build_url(self) -> str:
        now = int(time.time())
        params = {
            "secretid": self.secret_id,
            "timestamp": now,
            "expired": now + 24 * 60 * 60,
            "nonce": random.randint(1, 9_999_999_999),
            "engine_model_type": self.engine_model_type,
            "voice_id": str(uuid.uuid4()),
            "voice_format": _VOICE_FORMAT_PCM,
            "needvad": 1,
            "filter_dirty": 0,
            "filter_modal": self.filter_modal,
            # Keep Chinese digits like the v1 run (convert_num_mode=0); the
            # metric doesn't normalize number scripts.
            "convert_num_mode": 0,
        }
        for key in (
            "vad_silence_time",
            "filter_dirty",
            "filter_modal",
            "filter_punc",
            "convert_num_mode",
            "filter_empty_result",
            "hotword_id",
            "hotword_list",
        ):
            if key in self.options:
                params[key] = self.options[key]
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sign_str = f"{_HOST_PATH}{self.app_id}?{query}"
        digest = hmac.new(
            self.secret_key.encode(), sign_str.encode(), hashlib.sha1
        ).digest()
        signature = urllib.parse.quote(base64.b64encode(digest).decode(), safe="")
        return f"wss://{_HOST_PATH}{self.app_id}?{urllib.parse.urlencode(params)}&signature={signature}"

    async def recognize(
        self, audio_data: bytes, audio_format: str, sample_rate: int
    ) -> ASRResult:
        from ..streaming import Transcript

        if audio_format != "pcm" or sample_rate != 16000:
            raise ValueError("Tencent expects normalized 16kHz PCM")
        t0 = time.monotonic()
        transcript = Transcript()
        async with websockets.connect(
            self._build_url(), open_timeout=10, max_size=10_000_000
        ) as ws:
            handshake = json.loads(await asyncio.wait_for(ws.recv(), 10))
            if handshake.get("code") != 0:
                raise RuntimeError("Tencent handshake failed")

            async def sender():
                # Tencent requires 1:1 sending even when used in batch mode.
                start = time.monotonic()
                for offset in range(0, len(audio_data), 6400):
                    await asyncio.sleep(
                        max(0, start + offset / 32000 - time.monotonic())
                    )
                    await ws.send(audio_data[offset : offset + 6400])
                await asyncio.sleep(
                    max(0, start + len(audio_data) / 32000 - time.monotonic())
                )
                await ws.send(json.dumps({"type": "end"}))

            async def receiver():
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
                    if msg.get("code", 0) != 0:
                        raise RuntimeError("Tencent recognition failed")
                    transcript.update(msg, (time.monotonic() - t0) * 1000, True)
                    if msg.get("final") == 1:
                        return

            tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        return ASRResult(
            text=transcript.text, duration_ms=(time.monotonic() - t0) * 1000
        )
