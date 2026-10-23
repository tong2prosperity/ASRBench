import asyncio
import json

import pytest

from asrbench.providers.base import ASRResult
from asrbench.service import BenchService
from asrbench.summary import export_report


class FakeProvider:
    supported_formats = {"pcm"}

    def __init__(self, text="你好 world", fail=False):
        self.text, self.fail = text, fail

    async def recognize(self, data, fmt, sample_rate):
        assert fmt == "pcm" and sample_rate == 16000
        if self.fail:
            raise RuntimeError("test-only-secret must not appear")
        return ASRResult(self.text, 999)

    async def close(self):
        pass


def test_multi_profile_empty_output_and_exports(tmp_path, manifest, config):
    async def scenario():
        s = BenchService(
            config,
            tmp_path / "bench.db",
            provider_factory=lambda name, cfg: FakeProvider(
                "" if cfg["model"] == "fake-a" else "你好 world"
            ),
        )
        d = await s.import_dataset(*manifest)
        run = s.create_run(d["id"], ["alpha", "beta"])
        await s.schedule(run["id"])
        report = s.report(run["id"])
        assert report["state"] == "completed"
        assert report["common_count"] == 2
        assert report["profiles"][0]["accuracy"]["cer"] == 1
        assert report["profiles"][0]["success"] == 2
        assert report["profiles"][1]["accuracy"]["cer"] == 0
        assert "test-only" not in json.dumps(report)
        for fmt in ("csv", "json", "md"):
            content, media = export_report(
                report, s.store.attempts(run["id"], latest=False), fmt
            )
            assert content and media
        assert s.store.attempts(run["id"])[0]["request_elapsed_ms"] != 999
        await s.close()

    asyncio.run(scenario())


def test_failures_common_coverage_and_retry(tmp_path, manifest, config):
    async def scenario():
        failing = True

        def factory(name, cfg):
            return FakeProvider(fail=failing and cfg["model"] == "fake-b")

        s = BenchService(config, tmp_path / "bench.db", provider_factory=factory)
        d = await s.import_dataset(*manifest)
        run = s.create_run(d["id"], ["alpha", "beta"])
        rid = run["id"]
        await s.schedule(rid)
        r = s.report(rid)
        assert r["common_count"] == 0
        assert r["profiles"][1]["accuracy"]["cer"] is None
        assert r["profiles"][1]["failure_rate"] == 1
        assert "test-only-secret" not in str(s.store.attempts(rid))
        await s.schedule(rid, "resume")
        assert len(s.store.attempts(rid, latest=False)) == 4
        failing = False
        await s.schedule(rid, "retry")
        assert s.report(rid)["common_count"] == 2
        assert len(s.store.attempts(rid, latest=False)) == 6
        assert len(s.store.attempts(rid)) == 4
        await s.close()

    asyncio.run(scenario())


def test_stop_resume_and_process_exclusion(tmp_path, manifest, config):
    async def scenario():
        started = asyncio.Event()

        class Blocking(FakeProvider):
            async def recognize(self, *args):
                started.set()
                await asyncio.Event().wait()

        s = BenchService(
            config, tmp_path / "bench.db", provider_factory=lambda *args: Blocking()
        )
        d = await s.import_dataset(*manifest)
        rid = s.create_run(d["id"], ["alpha"], concurrency=1)["id"]
        task = s.schedule(rid)
        await started.wait()
        other = BenchService(config, tmp_path / "bench.db")
        with pytest.raises(ValueError, match="其他进程"):
            other.activate()
        assert s.store.run(rid)["state"] == "running"
        s.stop(rid)
        await task
        assert s.store.run(rid)["state"] == "cancelled"
        s.provider_factory = lambda *args: FakeProvider()
        await s.schedule(rid, "resume")
        assert s.report(rid)["profiles"][0]["success"] == 2
        await s.close()
        await other.close()

    asyncio.run(scenario())


def test_changed_audio_and_config_rejected(tmp_path, manifest, config):
    async def scenario():
        s = BenchService(config, tmp_path / "bench.db")
        d = await s.import_dataset(*manifest)
        rid = s.create_run(d["id"], ["alpha"])["id"]
        config["models"][0]["params"]["model"] = "changed"
        with pytest.raises(ValueError, match="模型配置已改变"):
            s.schedule(rid)
        config["models"][0]["params"]["model"] = "fake-a"
        (manifest[1] / "1.wav").write_bytes(b"changed")
        with pytest.raises(ValueError, match="音频已改变"):
            s.schedule(rid)
        await s.close()

    asyncio.run(scenario())


def test_stream_capability_and_invalid_concurrency(tmp_path, manifest, config):
    async def scenario():
        s = BenchService(config, tmp_path / "bench.db")
        d = await s.import_dataset(*manifest)
        with pytest.raises(ValueError, match="腾讯"):
            s.create_run(d["id"], ["alpha"], mode="stream")
        for n in (0, 33):
            with pytest.raises(ValueError, match="并发"):
                s.create_run(d["id"], ["alpha"], concurrency=n)
        await s.close()

    asyncio.run(scenario())


def test_recovery_and_stream_report(tmp_path, manifest, config):
    async def scenario():
        config["models"][0].update(
            provider="tencent",
            params={"app_id": "test", "secret_id": "test", "secret_key": "test"},
        )

        async def stream(name, params, pcm, emit):
            emit("sentence_final", {"text": "你好 world", "received_ms": 100})
            return dict(
                text="你好 world",
                final_latency_ms=-20,
                final_reason="音频结束前 final",
                first_text_latency_ms=20,
            )

        s = BenchService(config, tmp_path / "bench.db", stream_recognizer=stream)
        d = await s.import_dataset(*manifest)
        rid = s.create_run(d["id"], ["alpha"], mode="stream")["id"]
        s.store.state(rid, "running")
        s.activate()
        assert s.store.run(rid)["state"] == "interrupted"
        await s.schedule(rid, "resume")
        p = s.report(rid)["profiles"][0]
        assert p["final_latency_ms"]["mean"] == -20
        assert p["rtf"] is None
        assert p["final_coverage"] == 1
        await s.close()

    asyncio.run(scenario())


def test_timeout_and_elapsed_excludes_scoring(tmp_path, manifest, config, monkeypatch):
    import asrbench.service as module
    import time

    monkeypatch.setattr(module, "BATCH_TIMEOUT_S", 0.03)
    original = module.score_pair

    def slow_scoring(*args):
        time.sleep(0.06)
        return original(*args)

    monkeypatch.setattr(module, "score_pair", slow_scoring)

    class Slow(FakeProvider):
        async def recognize(self, *args):
            await asyncio.sleep(0.2)

    async def scenario():
        s = BenchService(
            config, tmp_path / "bench.db", provider_factory=lambda *args: Slow()
        )
        d = await s.import_dataset(*manifest)
        rid = s.create_run(d["id"], ["alpha"], concurrency=1)["id"]
        await s.schedule(rid)
        assert all(r["status"] == "timeout" for r in s.store.attempts(rid))
        s.provider_factory = lambda *args: FakeProvider()
        await s.schedule(rid, "retry")
        assert all(r["request_elapsed_ms"] < 50 for r in s.store.attempts(rid))
        assert s.report(rid)["profiles"][0]["success"] == 2
        await s.close()

    asyncio.run(scenario())


def test_stop_queued_task_is_immediate(tmp_path, manifest, config):
    async def scenario():
        started = asyncio.Event()

        class Blocking(FakeProvider):
            async def recognize(self, *args):
                started.set()
                await asyncio.Event().wait()

        s = BenchService(
            config, tmp_path / "bench.db", provider_factory=lambda *a: Blocking()
        )
        d = await s.import_dataset(*manifest)
        r1 = s.create_run(d["id"], ["alpha"])["id"]
        r2 = s.create_run(d["id"], ["beta"])["id"]
        t1 = s.schedule(r1)
        await started.wait()
        t2 = s.schedule(r2)
        await asyncio.sleep(0)
        s.stop(r2)
        await asyncio.gather(t2, return_exceptions=True)
        assert s.store.run(r2)["state"] == "cancelled"
        assert not s.store.attempts(r2)
        s.stop(r1)
        await t1
        await s.close()

    asyncio.run(scenario())
