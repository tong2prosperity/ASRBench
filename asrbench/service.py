"""Shared run lifecycle for the CLI and local HTTP service."""

import asyncio
import contextlib
import time
from pathlib import Path

from filelock import FileLock, Timeout

from .audio import prepare_pcm
from .config import STREAMING, public_profiles, redact, resolve_profile
from .dataset import validate_manifest
from .metrics import METRIC_VERSION, normalize_text, score_pair
from .store import Store
from .summary import summarize


BATCH_TIMEOUT_S = 60
STREAM_OVERHEAD_S = 40


class BenchService:
    def __init__(
        self,
        config,
        db="runtime/asrbench.sqlite3",
        provider_factory=None,
        stream_recognizer=None,
    ):
        self.config = config
        self.store = Store(db)
        self.process_lock = FileLock(self.store.path + ".lock")
        self.active = False
        self.lock = asyncio.Lock()
        self.jobs = {}
        self.stops = {}
        self.provider_factory = provider_factory
        self.stream_recognizer = stream_recognizer

    def activate(self):
        if self.active:
            return
        try:
            self.process_lock.acquire(timeout=0)
        except Timeout:
            raise ValueError(
                "此数据库正由其他进程执行任务；请使用已启动的网页，或指定其他 --db"
            ) from None
        self.store.recover()
        self.active = True

    async def import_dataset(self, path, audio_root=None):
        result = await asyncio.to_thread(validate_manifest, path, audio_root)
        # Punctuation-only speech references cannot have a meaningful CER denominator.
        for sample in result["samples"]:
            if not "".join(normalize_text(sample["ground_truth"]).split()):
                result["errors"].append(
                    {"id": sample["id"], "error": "参考文本规范化后为空"}
                )
        body = dict(
            result,
            manifest=str(Path(path).expanduser().resolve()),
            audio_root=str(Path(audio_root).expanduser().resolve())
            if audio_root
            else None,
        )
        return self.store.add_dataset(body)

    def profiles(self):
        return public_profiles(self.config)

    def _current_profiles(self, run):
        current = {
            p["profile_id"]: p for p in self.config["models"] if p.get("enabled", True)
        }
        output = []
        for snapshot in run["spec"]["profiles"]:
            p = current.get(snapshot["profile_id"])
            if p is None or redact(p) != snapshot:
                raise ValueError("模型配置已改变，请新建任务")
            output.append(p)
        return output

    def _verify_audio(self, samples):
        for sample in samples:
            try:
                s = Path(sample["audio_path"]).stat()
            except OSError:
                raise ValueError("音频缺失，请重新导入数据集") from None
            m = sample["metadata"]
            if s.st_size != m["size"] or s.st_mtime_ns != m["mtime_ns"]:
                raise ValueError("音频已改变，请重新导入并新建任务")

    def create_run(
        self, dataset_id, profile_ids, mode="batch", concurrency=None, limit=None
    ):
        dataset = self.store.dataset(dataset_id)
        if dataset["errors"]:
            raise ValueError("数据集有校验错误，请修正后重新导入")
        if mode not in ("batch", "stream"):
            raise ValueError("mode 必须为 batch 或 stream")
        if not profile_ids or len(profile_ids) != len(set(profile_ids)):
            raise ValueError("请选择不重复的模型配置")
        lookup = {
            p["profile_id"]: p for p in self.config["models"] if p.get("enabled", True)
        }
        if any(pid not in lookup for pid in profile_ids):
            raise ValueError("模型配置不存在或未启用")
        profiles = [lookup[pid] for pid in profile_ids]
        if mode == "stream" and any(p["provider"] not in STREAMING for p in profiles):
            raise ValueError("一期自然 final 延迟只支持腾讯两套适配器")
        concurrency = (
            concurrency if concurrency is not None else (1 if mode == "stream" else 3)
        )
        if (
            not isinstance(concurrency, int)
            or not 1 <= concurrency <= 32
            or (mode == "stream" and concurrency != 1)
        ):
            raise ValueError("批测并发必须为 1–32；流式并发固定为 1")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("样本数量必须为正整数")
        samples = dataset["samples"][:limit] if limit else dataset["samples"]
        if not samples:
            raise ValueError("没有可评测样本")
        self._verify_audio(samples)
        for p in profiles:
            params = resolve_profile(p)
            required = {
                "tencent": ("app_id", "secret_id", "secret_key"),
                "tencent_v2": ("app_id", "secret_id", "secret_key"),
                "volcengine": ("app_key", "access_key"),
            }.get(p["provider"], ("api_key",))
            if any(not params.get(k) for k in required):
                raise ValueError(
                    f"模型 {p['profile_id']} 缺少凭据参数：{', '.join(required)}"
                )
        return self.store.create_run(
            {
                "dataset_id": dataset_id,
                "samples": samples,
                "profiles": redact(profiles),
                "mode": mode,
                "concurrency": concurrency,
                "metric_version": METRIC_VERSION,
                "transport_policy": {
                    p["profile_id"]: "tencent_1to1_200ms"
                    if p["provider"] in STREAMING
                    else "provider_batch"
                    for p in profiles
                },
                "silence_seconds": 10 if mode == "stream" else 0,
                "chunk_ms": 200 if mode == "stream" else None,
            }
        )

    def schedule(self, rid, action="run"):
        self.activate()
        run = self.store.run(rid)
        if rid in self.jobs or run["state"] in ("running", "stopping"):
            raise ValueError("任务正在执行或等待执行")
        if action not in ("run", "resume", "retry"):
            raise ValueError("未知任务操作")
        self._verify_audio(run["spec"]["samples"])
        self._current_profiles(run)
        self.stops[rid] = asyncio.Event()
        self.store.state(rid, "queued")
        task = asyncio.create_task(self._execute(rid, action))
        self.jobs[rid] = task

        def finished(_):
            self.jobs.pop(rid, None)
            self.stops.pop(rid, None)

        task.add_done_callback(finished)
        return task

    def stop(self, rid):
        run = self.store.run(rid)
        if rid not in self.jobs:
            return run
        self.stops[rid].set()
        if run["state"] == "queued":
            self.store.state(rid, "cancelled")
            self.jobs[rid].cancel()
        else:
            self.store.state(rid, "stopping")
        return self.store.run(rid)

    def report(self, rid):
        return summarize(self.store.run(rid), self.store.attempts(rid))

    async def close(self):
        for stop in self.stops.values():
            stop.set()
        tasks = list(self.jobs.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.active:
            self.process_lock.release()
            self.active = False

    async def _execute(self, rid, action):
        stop = self.stops[rid]
        try:
            async with self.lock:
                run = self.store.run(rid)
                self._verify_audio(run["spec"]["samples"])
                profiles = self._current_profiles(run)
                if stop.is_set():
                    self.store.state(rid, "cancelled")
                    return
                self.store.state(rid, "running")
                previous = {
                    (r["profile_id"], r["sample_id"]): r
                    for r in self.store.attempts(rid)
                }
                for profile in profiles:
                    if stop.is_set():
                        break
                    params = resolve_profile(profile)
                    sem = asyncio.Semaphore(run["spec"]["concurrency"])

                    async def process(sample):
                        pid, sid = profile["profile_id"], sample["id"]
                        old = previous.get((pid, sid))
                        # Resume only unfinished/cancelled work. Retry only failed requests.
                        if old and (
                            old["status"] == "success"
                            or (
                                action != "retry"
                                and old["status"] in ("failed", "timeout")
                            )
                        ):
                            return
                        if action == "retry" and (
                            not old or old["status"] not in ("failed", "timeout")
                        ):
                            return
                        async with sem:
                            if stop.is_set():
                                return
                            await self._process(
                                rid, profile, params, sample, run["spec"]["mode"], stop
                            )

                    # Bounded batches avoid allocating a task for every sample in large datasets.
                    samples = run["spec"]["samples"]
                    batch_size = run["spec"]["concurrency"] * 2
                    for i in range(0, len(samples), batch_size):
                        if stop.is_set():
                            break
                        await asyncio.gather(
                            *(process(s) for s in samples[i : i + batch_size])
                        )
                self.store.state(rid, "cancelled" if stop.is_set() else "completed")
        except asyncio.CancelledError:
            if self.store.run(rid)["state"] != "cancelled":
                self.store.state(rid, "interrupted")
            raise
        except Exception as exc:
            self.store.event(
                rid,
                "error",
                {"error": f"任务异常：{type(exc).__name__}；请检查本地配置及音频"},
            )
            self.store.state(rid, "failed")
        finally:
            self.stops.pop(rid, None)

    async def _process(self, rid, profile, params, sample, mode, stop):
        pid, sid = profile["profile_id"], sample["id"]
        body = {
            "status": "failed",
            "reference_text": sample["ground_truth"],
            "text": "",
            "audio_duration_ms": sample["audio_duration_ms"],
            "request_elapsed_ms": None,
            "cer": None,
            "wer": None,
            "final_latency_ms": None,
        }
        self.store.event(rid, "sample_start", {"profile_id": pid, "sample_id": sid})
        provider = None
        began = None
        work = stop_task = None
        try:
            pcm, duration = await asyncio.to_thread(prepare_pcm, sample["audio_path"])
            body["audio_duration_ms"] = duration

            def emit(kind, payload):
                self.store.event(
                    rid, kind, dict(payload, profile_id=pid, sample_id=sid)
                )

            if mode == "stream":
                from .streaming import recognize_stream

                fn = self.stream_recognizer or recognize_stream
                call = fn(profile["provider"], params, pcm, emit)
                timeout = duration / 1000 + STREAM_OVERHEAD_S
            else:
                if self.provider_factory:
                    provider = self.provider_factory(profile["provider"], params)
                else:
                    from .providers import PROVIDER_REGISTRY

                    provider = PROVIDER_REGISTRY[profile["provider"]](params)
                # Normalize once; providers accepting files receive a normalized WAV.
                data, fmt = pcm, "pcm"
                if "pcm" not in provider.supported_formats:
                    from .audio import _pcm_to_wav

                    data, fmt = _pcm_to_wav(pcm, 16000), "wav"
                call = provider.recognize(data, fmt, 16000)
                timeout = BATCH_TIMEOUT_S
            began = time.monotonic()

            async def timed_call():
                try:
                    return await call
                finally:
                    body["request_elapsed_ms"] = (time.monotonic() - began) * 1000

            work = asyncio.create_task(timed_call())
            stop_task = asyncio.create_task(stop.wait())
            done, _ = await asyncio.wait(
                [work, stop_task], timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if stop_task in done:
                body.update(status="cancelled", error="用户停止任务")
            elif work not in done:
                body.update(status="timeout", error="请求超时")
            else:
                result = await work
                if mode == "stream":
                    body.update(result)
                else:
                    raw = result.raw_response
                    if isinstance(raw, dict) and raw.get("error"):
                        raise RuntimeError("供应商返回识别错误")
                    body["text"] = result.text
                body.update(
                    score_pair(body["reference_text"], body["text"]), status="success"
                )
        except asyncio.CancelledError:
            body.update(status="cancelled", error="服务中断")
            raise
        except (TimeoutError, asyncio.TimeoutError):
            body.update(status="timeout", error="请求超时")
        except Exception as exc:
            body.update(
                status="failed",
                error=f"{type(exc).__name__}：识别失败，请检查凭据、模型权限、配额或连接",
            )
        finally:
            if began is not None and body["request_elapsed_ms"] is None:
                body["request_elapsed_ms"] = (time.monotonic() - began) * 1000
            for task in (work, stop_task):
                if task:
                    task.cancel()
            for task in (work, stop_task):
                if task:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            if provider:
                with contextlib.suppress(Exception):
                    await provider.close()
            self.store.attempt(rid, pid, sid, body)
