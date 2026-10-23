import asyncio
import json

import pytest

from asrbench.streaming import Transcript, replay_socket


def packet(text, start=0, kind=2):
    return {
        "code": 0,
        "result": {
            "slice_type": kind,
            "index": 0,
            "start_time": start,
            "voice_text_str": text,
        },
    }


def test_segments_repeated_ids_revisions_and_order():
    t = Transcript()
    t.update(packet("后", 100), 180)
    t.update(packet("前", 0), 190)
    t.update(packet("后修订", 100), 200)
    t.update(packet("", 200), 210)
    assert t.text == "前后修订"
    assert t.latency(200, 0) == (0, "自然 final")


def test_sentences_protocol_forced_and_missing():
    t = Transcript()
    t.update(
        {"sentences": {"sentence_id": 0, "sentence": "你好", "sentence_type": 1}}, 30
    )
    assert t.latency(100, 0) == (-70, "音频结束前 final")
    t.update(
        {"sentences": {"sentence_id": 0, "sentence": "您好", "sentence_type": 1}},
        140,
        True,
    )
    assert t.text == "您好"
    assert t.latency(100, 0)[0] is None
    assert t.latency(100, 101)[1] == "发送时钟偏差超过 100ms"
    assert Transcript().latency(100, 0)[0] is None


def test_session_complete_is_not_sentence_final():
    t = Transcript()
    assert t.update({"final": 1}, 100) is None
    assert t.latency(100, 0)[0] is None


def test_paced_frames_observation_and_end():
    async def scenario():
        now = [10.0]
        writes, events = [], []
        queue = asyncio.Queue()

        class Socket:
            async def send(self, data):
                writes.append((now[0], data))
                if isinstance(data, bytes) and len(writes) == 1:
                    queue.put_nowait(json.dumps(packet("你好")))
                if isinstance(data, str):
                    queue.put_nowait(json.dumps({"final": 1}))
                await asyncio.sleep(0)

            async def recv(self):
                return await queue.get()

        async def sleep(delay):
            now[0] += delay
            await asyncio.sleep(0)

        result = await replay_socket(
            Socket(),
            bytes(9600),
            lambda k, p: events.append((k, p)),
            clock=lambda: now[0],
            sleep=sleep,
        )
        assert result["text"] == "你好"
        assert result["final_latency_ms"] < 0
        assert writes[-1][0] == pytest.approx(20.3)
        assert json.loads(writes[-1][1]) == {"type": "end"}
        assert all(
            t2 - t1 == pytest.approx(0.2)
            for (t1, d1), (t2, d2) in zip(writes[:2], writes[1:3])
        )
        assert (
            sum(len(data) for _, data in writes if isinstance(data, bytes))
            == 9600 + 320000
        )
        assert any(k == "session_complete" for k, p in events)

    asyncio.run(scenario())


def test_forced_final_does_not_count():
    async def scenario():
        q = asyncio.Queue()

        class Socket:
            async def send(self, data):
                if isinstance(data, str):
                    q.put_nowait(json.dumps(packet("你好")))
                    q.put_nowait(json.dumps({"final": 1}))

            async def recv(self):
                return await q.get()

        result = await replay_socket(
            Socket(), bytes(640), lambda *args: None, silence_s=0.01, chunk_ms=10
        )
        assert result["text"] == "你好"
        assert result["final_latency_ms"] is None
        assert result["final_reason"] == "结束输入后返回"

    asyncio.run(scenario())


def test_receiver_failure_cleans_up_sender():
    async def scenario():
        class Socket:
            async def send(self, data):
                pass

            async def recv(self):
                return json.dumps({"code": 123})

        with pytest.raises(RuntimeError):
            await replay_socket(Socket(), bytes(32000), lambda *args: None)
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    asyncio.run(scenario())
