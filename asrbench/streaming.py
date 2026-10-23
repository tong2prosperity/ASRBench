"""Tencent real-time replay. Session completion is never a sentence final."""

import asyncio
import contextlib
import json
import time

import websockets


class Transcript:
    def __init__(self):
        self.segments = {}
        self.order = []
        self.active = None
        self.counter = 0
        self.messages = set()

    def update(self, msg, received_ms, forced=False):
        mid = msg.get("message_id")
        if mid and mid in self.messages:
            return None
        if mid:
            self.messages.add(mid)
        r = msg.get("result")
        if isinstance(r, dict):
            text = r.get("voice_text_str")
            final = r.get("slice_type") == 2
            begin = r.get("slice_type") == 0
            identity = r.get("index")
        else:
            r = msg.get("sentences")
            if not isinstance(r, dict):
                return None
            text = r.get("text", r.get("sentence"))
            final = r.get("sentence_type") == 1
            begin = False
            identity = r.get("sentence_id")
        if text is None:
            return None
        # start_time also handles observed engines that reuse index=0.
        if r.get("start_time") is not None:
            key = ("start", r["start_time"])
        elif isinstance(msg.get("sentences"), dict) and identity is not None:
            key = ("sentence", identity)
        elif begin or self.active is None:
            self.counter += 1
            key = ("sequence", self.counter)
        else:
            key = self.active
            old = self.segments[key]
            if old.get("identity") != identity or (
                old["final"] and text != old["text"]
            ):
                self.counter += 1
                key = ("sequence", self.counter)
        if key not in self.segments:
            self.order.append(key)
        self.active = key
        self.segments[key] = {
            "text": text,
            "final": final,
            "received_ms": received_ms,
            "forced": forced,
            "identity": identity,
            "start_ms": r.get("start_time"),
        }
        return {
            "segment_id": str(key),
            "text": text,
            "final": final,
            "received_ms": received_ms,
            "forced": forced,
        }

    def ordered(self):
        indexed = list(enumerate(self.segments[k] for k in self.order))
        return [
            s
            for i, s in sorted(
                indexed,
                key=lambda pair: (
                    pair[1]["start_ms"] if pair[1]["start_ms"] is not None else pair[0]
                ),
            )
        ]

    @property
    def text(self):
        return "".join(s["text"] for s in self.ordered())

    def latency(self, audio_ms, drift_ms):
        effective = [s for s in self.ordered() if s["text"].strip()]
        if drift_ms > 100:
            return None, "发送时钟偏差超过 100ms"
        if not effective:
            return None, "未返回有效文本"
        last = effective[-1]
        if not last["final"]:
            return None, "最后有效句段没有 final"
        if last["forced"]:
            return None, "结束输入后返回"
        value = last["received_ms"] - audio_ms
        return value, "音频结束前 final" if value < 0 else "自然 final"


async def replay_socket(
    ws,
    pcm,
    emit,
    *,
    silence_s=10,
    chunk_ms=200,
    clock=time.monotonic,
    sleep=asyncio.sleep,
):
    """Socket has already completed the recognition handshake."""
    transcript = Transcript()
    audio_ms = len(pcm) / 32.0
    start = clock()
    end_sent = False
    max_drift = 0.0
    first_text = None

    async def sender():
        nonlocal end_sent, max_drift, start
        payload = pcm + bytes(round(silence_s * 16000) * 2)
        chunk_bytes = chunk_ms * 32
        start = clock()
        for offset in range(0, len(payload), chunk_bytes):
            due = start + offset / 32000
            await sleep(max(0, due - clock()))
            await ws.send(payload[offset : offset + chunk_bytes])
            drift = abs(clock() - due) * 1000
            max_drift = max(max_drift, drift)
            emit(
                "audio_frame",
                {
                    "offset_ms": offset / 32,
                    "elapsed_ms": (clock() - start) * 1000,
                    "drift_ms": drift,
                },
            )
        await sleep(max(0, start + len(payload) / 32000 - clock()))
        # Flag before writing: callbacks racing with end must be classified conservatively.
        end_sent = True
        emit("end_input", {"elapsed_ms": (clock() - start) * 1000})
        await ws.send(json.dumps({"type": "end"}))

    async def receiver():
        nonlocal first_text
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("code", 0) != 0:
                emit("provider_error", {"code": msg.get("code")})
                raise RuntimeError("腾讯返回错误，请检查凭据、配额和模型权限")
            received = (clock() - start) * 1000
            event = transcript.update(msg, received, end_sent)
            if event:
                emit("sentence_final" if event["final"] else "partial", event)
                if event["text"].strip() and first_text is None:
                    first_text = received
            if msg.get("final") == 1:
                emit("session_complete", {"elapsed_ms": received, "forced": end_sent})
                if not end_sent:
                    raise RuntimeError("服务在观察窗口结束前关闭会话")
                return

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(receiver())
    try:
        await asyncio.gather(send_task, recv_task)
    finally:
        for task in (send_task, recv_task):
            task.cancel()
        for task in (send_task, recv_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    latency, reason = transcript.latency(audio_ms, max_drift)
    return {
        "text": transcript.text,
        "final_latency_ms": latency,
        "final_reason": reason,
        "first_text_latency_ms": first_text,
        "max_send_drift_ms": max_drift,
    }


async def recognize_stream(provider_name, params, pcm, emit):
    # Both adapters use the same Tencent socket protocol; the engine is configurable.
    from .providers.tencent_v2 import TencentV2Provider

    config = dict(params)
    config.setdefault(
        "engine_model_type",
        "16k_zh_en_2.0",
    )
    provider = TencentV2Provider(config)
    async with websockets.connect(
        provider._build_url(), open_timeout=10, max_size=10_000_000
    ) as ws:
        handshake = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if handshake.get("code") != 0:
            emit("provider_error", {"code": handshake.get("code")})
            raise RuntimeError("腾讯握手失败，请检查凭据和模型权限")
        emit("ready", {})
        return await replay_socket(ws, pcm, emit)
