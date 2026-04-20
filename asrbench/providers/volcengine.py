"""VolcEngine streaming ASR 大模型版 (v3) provider.

Docs: https://www.volcengine.com/docs/6561/1354869

Uses the `bigmodel_nostream` endpoint (streaming input, single recognition result)
which is best suited for short offline clips like Common Voice samples.
"""
import asyncio
import gzip
import json
import struct
import time
import uuid

import websockets

from .base import ASRProvider, ASRResult

# Message types
_FULL_CLIENT_REQUEST = 0b0001
_AUDIO_ONLY_REQUEST = 0b0010
_FULL_SERVER_RESPONSE = 0b1001
_SERVER_ACK = 0b1011
_SERVER_ERROR_RESPONSE = 0b1111

# Message type specific flags
_POS_SEQUENCE = 0b0001       # carries positive sequence number
_NEG_WITH_SEQUENCE = 0b0011  # last packet, carries negative sequence number

# Serialization + compression
_JSON = 0b0001
_GZIP = 0b0001
_NONE = 0b0000


def _header(message_type: int, flags: int, serial: int = _JSON, compression: int = _GZIP) -> bytearray:
    h = bytearray(4)
    h[0] = (0b0001 << 4) | 0b0001  # version=1, header_size=1 (4 bytes)
    h[1] = (message_type << 4) | flags
    h[2] = (serial << 4) | compression
    h[3] = 0x00
    return h


def _parse_response(data: bytes) -> dict:
    header_size = data[0] & 0x0F
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    serial = data[2] >> 4
    compression = data[2] & 0x0F
    payload = data[header_size * 4:]

    result: dict = {"is_last": False}
    if flags & 0x01:
        result["sequence"] = struct.unpack(">i", payload[:4])[0]
        payload = payload[4:]
    if flags & 0x02:
        result["is_last"] = True

    payload_msg = None
    if message_type == _FULL_SERVER_RESPONSE:
        size = struct.unpack(">I", payload[:4])[0]
        payload_msg = payload[4:4 + size] if size else payload[4:]
    elif message_type == _SERVER_ACK:
        if len(payload) >= 8:
            size = struct.unpack(">I", payload[4:8])[0]
            payload_msg = payload[8:8 + size] if size else payload[8:]
    elif message_type == _SERVER_ERROR_RESPONSE:
        result["code"] = struct.unpack(">I", payload[:4])[0]
        size = struct.unpack(">I", payload[4:8])[0]
        payload_msg = payload[8:8 + size] if size else payload[8:]

    if payload_msg:
        if compression == _GZIP:
            payload_msg = gzip.decompress(payload_msg)
        if serial == _JSON:
            try:
                payload_msg = json.loads(payload_msg.decode("utf-8"))
            except Exception:
                payload_msg = payload_msg.decode("utf-8", errors="replace")
        result["message"] = payload_msg
    return result


def _extract_text(message) -> str:
    if not isinstance(message, dict):
        return ""
    res = message.get("result")
    if isinstance(res, dict):
        return res.get("text", "") or ""
    if isinstance(res, list):
        return "".join(item.get("text", "") for item in res if isinstance(item, dict))
    return ""


class VolcEngineProvider(ASRProvider):
    name = "volcengine"
    supported_formats = frozenset({"wav", "pcm", "mp3"})

    def __init__(self, config: dict):
        self.app_key = str(config.get("app_key") or config.get("appid") or "")
        self.access_key = config.get("access_key") or config.get("token") or ""
        self.resource_id = config.get("resource_id", "volc.bigasr.sauc.duration")
        self.ws_url = config.get(
            "ws_url", "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_nostream"
        )
        # 200ms chunks recommended by docs
        self.chunk_ms = config.get("chunk_ms", 200)

    def _build_request(self, audio_format: str, sample_rate: int) -> dict:
        return {
            "user": {"uid": "asrbench"},
            "audio": {
                "format": "pcm" if audio_format == "pcm" else audio_format,
                "codec": "raw",
                "rate": sample_rate,
                "bits": 16,
                "channel": 1,
                "language": "zh-CN",
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": False,
                "enable_punc": True,
                "enable_ddc": False,
                "show_utterances": False,
                "result_type": "full",
            },
        }

    async def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int) -> ASRResult:
        t0 = time.monotonic()
        headers = {
            "X-Api-App-Key": self.app_key,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

        # Chunk size in bytes for 16-bit mono PCM.
        bytes_per_ms = sample_rate * 2 // 1000  # 16-bit mono
        if audio_format in ("pcm", "wav"):
            chunk_size = max(bytes_per_ms * self.chunk_ms, 3200)
        else:  # mp3
            chunk_size = 8000  # ~200ms at 32kbps

        request_params = self._build_request(audio_format, sample_rate)
        payload = gzip.compress(json.dumps(request_params).encode("utf-8"))

        sequence = 1
        full_req = _header(_FULL_CLIENT_REQUEST, _POS_SEQUENCE)
        full_req.extend(struct.pack(">i", sequence))
        full_req.extend(struct.pack(">I", len(payload)))
        full_req.extend(payload)

        last_message = None
        try:
            async with websockets.connect(
                self.ws_url, additional_headers=headers, max_size=10_000_000
            ) as ws:
                await ws.send(bytes(full_req))
                resp = _parse_response(await ws.recv())
                last_message = resp
                if "code" in resp:
                    elapsed = (time.monotonic() - t0) * 1000
                    return ASRResult(text="", duration_ms=elapsed, raw_response=resp)

                offset = 0
                total = len(audio_data)
                while offset < total:
                    chunk = audio_data[offset:offset + chunk_size]
                    offset += chunk_size
                    is_last = offset >= total
                    sequence += 1
                    if is_last:
                        seq_to_send = -sequence
                        flags = _NEG_WITH_SEQUENCE
                    else:
                        seq_to_send = sequence
                        flags = _POS_SEQUENCE
                    compressed = gzip.compress(chunk)
                    audio_req = _header(_AUDIO_ONLY_REQUEST, flags)
                    audio_req.extend(struct.pack(">i", seq_to_send))
                    audio_req.extend(struct.pack(">I", len(compressed)))
                    audio_req.extend(compressed)
                    await ws.send(bytes(audio_req))
                    resp = _parse_response(await ws.recv())
                    last_message = resp
                    if "code" in resp:
                        break

                # Drain any remaining messages until server signals last.
                while not last_message.get("is_last") and "code" not in last_message:
                    try:
                        resp = _parse_response(await asyncio.wait_for(ws.recv(), timeout=5))
                        last_message = resp
                    except asyncio.TimeoutError:
                        break
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return ASRResult(text="", duration_ms=elapsed, raw_response={"error": str(e)})

        elapsed = (time.monotonic() - t0) * 1000
        text = _extract_text(last_message.get("message") if last_message else None)
        return ASRResult(text=text, duration_ms=elapsed, raw_response=last_message or {})
