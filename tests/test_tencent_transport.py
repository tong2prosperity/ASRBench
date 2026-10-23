import asyncio
import json

import pytest
import websockets

from asrbench.providers.tencent import TencentProvider
from asrbench.providers.tencent_v2 import TencentV2Provider


@pytest.mark.parametrize("provider_class", [TencentProvider, TencentV2Provider])
def test_batch_transport_handshake_frames_duplicates_end(monkeypatch, provider_class):
    async def scenario():
        packets = []

        async def server(ws):
            await ws.send(json.dumps({"code": 0, "voice_id": "synthetic"}))
            async for message in ws:
                packets.append(message)
                if isinstance(message, bytes):
                    if provider_class is TencentProvider:
                        msg = {
                            "code": 0,
                            "result": {
                                "index": 0,
                                "start_time": 0,
                                "slice_type": 2,
                                "voice_text_str": "你好",
                            },
                        }
                    else:
                        msg = {
                            "code": 0,
                            "sentences": {
                                "sentence_id": 0,
                                "start_time": 0,
                                "sentence_type": 1,
                                "sentence": "你好",
                            },
                        }
                    await ws.send(json.dumps(msg))
                    await ws.send(json.dumps(msg))
                else:
                    assert json.loads(message) == {"type": "end"}
                    await ws.send(json.dumps({"code": 0, "final": 1}))
                    return

        async with websockets.serve(server, "127.0.0.1", 0) as local:
            port = local.sockets[0].getsockname()[1]
            monkeypatch.setattr(
                TencentV2Provider, "_build_url", lambda self: f"ws://127.0.0.1:{port}"
            )
            p = provider_class(
                {
                    "app_id": "synthetic",
                    "secret_id": "synthetic",
                    "secret_key": "synthetic",
                }
            )
            result = await p.recognize(bytes(640), "pcm", 16000)
            assert result.text == "你好"
            assert result.duration_ms >= 20
            assert packets[0] == bytes(640)
            assert len(packets) == 2

    asyncio.run(scenario())


def test_all_tencent_defaults_are_v2():
    from pathlib import Path
    from asrbench.config import load_config

    credentials = {
        "app_id": "synthetic",
        "secret_id": "synthetic",
        "secret_key": "synthetic",
    }
    assert TencentProvider(credentials).engine_model_type == "16k_zh_en_2.0"
    assert TencentV2Provider(credentials).engine_model_type == "16k_zh_en_2.0"
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml.example")
    profiles = [p for p in cfg["models"] if p["provider"] in ("tencent", "tencent_v2")]
    assert len(profiles) == 1
    assert profiles[0]["params"]["engine_model_type"] == "16k_zh_en_2.0"
