"""Tencent Cloud real-time ASR V2 (WebSocket) — `16k_zh_en_2.0` engine.

Docs: https://cloud.tencent.com/document/api/1093/131127

Same host/path and HMAC-SHA1 signing as the v1 streaming API
(`wss://asr.cloud.tencent.com/asr/v2/<appid>`), but the response schema is
different: results arrive as `sentences: {sentence_id, sentence_type, text,
...}` instead of `result.slice_type / voice_text_str`, so the vendored
SpeechRecognizer can't parse it. This provider talks to the socket directly.

Sending pattern mirrors TencentProvider: 16k mono PCM in 6400-byte (200ms)
chunks written back-to-back, then `{"type": "end"}`, then wait for final=1.
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
        self.app_id = str(config["app_id"])
        self.secret_id = config["secret_id"]
        self.secret_key = config["secret_key"]
        self.engine_model_type = config.get("engine_model_type", "16k_zh_en_2.0")
        self.slice_size = config.get("slice_size", 6400)
        # Seconds to sleep between chunks; 0 matches the v1 tencent run.
        self.send_interval = config.get("send_interval", 0.0)
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
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sign_str = f"{_HOST_PATH}{self.app_id}?{query}"
        digest = hmac.new(self.secret_key.encode(), sign_str.encode(), hashlib.sha1).digest()
        signature = urllib.parse.quote(base64.b64encode(digest).decode(), safe="")
        return f"wss://{sign_str}&signature={signature}"

    async def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        t0 = time.monotonic()
        # Segments in arrival (= audio) order as [text, finalized]. The 2.0
        # engine reports every segment with index=0, so index can't be used as
        # a key: a trailing empty segment (e.g. a filtered-out "嗯") would
        # overwrite the real sentence. Instead a message opens a new segment
        # when it starts one (slice_type 0) or the previous one is already
        # final (slice_type 2); otherwise it updates the current segment.
        segments: list[list] = []
        voice_id = None
        messages = 0
        try:
            async with websockets.connect(self._build_url(), max_size=None) as ws:
                handshake = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
                voice_id = handshake.get("voice_id")
                if handshake.get("code") != 0:
                    return self._fail(t0, handshake, voice_id)

                async def sender():
                    for off in range(0, len(audio_data), self.slice_size):
                        await ws.send(audio_data[off:off + self.slice_size])
                        if self.send_interval:
                            await asyncio.sleep(self.send_interval)
                    await ws.send(json.dumps({"type": "end"}))

                send_task = asyncio.create_task(sender())
                try:
                    while True:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), self.timeout))
                        messages += 1
                        if msg.get("code") != 0:
                            return self._fail(t0, msg, voice_id)
                        # Live service (2026-09) still answers with the v1
                        # `result` shape; the documented `sentences` shape is
                        # handled too in case it switches.
                        r = msg.get("result")
                        if isinstance(r, dict) and r.get("voice_text_str") is not None:
                            slice_type = r.get("slice_type")
                            if slice_type == 0 or not segments or segments[-1][1]:
                                segments.append(["", False])
                            segments[-1] = [r["voice_text_str"], slice_type == 2]
                        s = msg.get("sentences")
                        if isinstance(s, dict):
                            text = s.get("text") if s.get("text") is not None else s.get("sentence")
                            if text is not None:
                                if not segments or segments[-1][1]:
                                    segments.append(["", False])
                                segments[-1] = [text, s.get("sentence_type") == 1]
                        if msg.get("final") == 1:
                            break
                finally:
                    send_task.cancel()
        except Exception as e:
            return ASRResult(
                text="",
                duration_ms=(time.monotonic() - t0) * 1000,
                raw_response={"error": f"{type(e).__name__}: {e}", "voice_id": voice_id},
            )

        text = "".join(t for t, _ in segments)
        return ASRResult(
            text=text,
            duration_ms=(time.monotonic() - t0) * 1000,
            raw_response={"voice_id": voice_id, "segments": len(segments), "messages": messages},
        )

    @staticmethod
    def _fail(t0, msg, voice_id):
        return ASRResult(
            text="",
            duration_ms=(time.monotonic() - t0) * 1000,
            raw_response={"error": f"{msg.get('code')}: {msg.get('message')}", "voice_id": voice_id},
        )
