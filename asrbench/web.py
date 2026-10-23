"""Local-only HTTP interface; all execution stays in BenchService."""

import asyncio
import difflib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from .service import BenchService
from .store import uid
from .summary import export_report


class ImportRequest(BaseModel):
    kind: str = "asr"
    manifest_path: str | None = None
    manifest_text: str | None = None
    audio_root: str | None = None


class RunRequest(BaseModel):
    dataset_id: str
    profile_ids: list[str]
    mode: str = "batch"
    concurrency: int | None = Field(default=None, ge=1, le=32)
    limit: int | None = Field(default=None, ge=1)
    prompt_template: str | None = None
    system_prompt: str | None = None
    label_policy: str | None = None


def create_app(config=None, db="runtime/asrbench.sqlite3", service=None):
    service = service or BenchService(config, db)

    @asynccontextmanager
    async def lifespan(app):
        service.activate()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="ASRBench", lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(OSError)
    async def unavailable(request, exc):
        return JSONResponse(
            {"detail": "无法读取本地文件，请检查路径和权限"}, status_code=400
        )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (Path(__file__).parent / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    @app.get("/api/models")
    async def models():
        return service.profiles()

    @app.get("/api/datasets")
    async def datasets():
        return [
            dict(
                id=d["id"],
                kind=d.get("kind", "asr"),
                count=d["count"],
                manifest=d["manifest"],
                errors=d["errors"],
                warnings=d["warnings"],
            )
            for d in service.store.datasets()
        ]

    @app.post("/api/datasets")
    async def import_dataset(body: ImportRequest):
        if (body.manifest_path is None) == (body.manifest_text is None):
            raise ValueError("必须选择文件内容或本地路径其中一种")
        path = body.manifest_path
        if body.manifest_text is not None:
            if len(body.manifest_text.encode("utf-8")) > 20 * 1024 * 1024:
                raise ValueError("清单超过 20MB，请使用本地路径导入")
            dest = (
                Path(service.store.path).parent
                / "imports"
                / (uid("manifest") + ".jsonl")
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body.manifest_text, encoding="utf-8")
            path = str(dest)
        return await service.import_dataset(path, body.audio_root, body.kind)

    @app.get("/api/eot/defaults")
    async def eot_defaults():
        return service.eot_defaults()

    @app.post("/api/datasets/builtin-eot")
    async def builtin_eot():
        return await service.import_builtin_eot()

    @app.get("/api/datasets/{did}")
    async def dataset(did: str):
        return service.store.dataset(did)

    @app.get("/api/runs")
    async def runs():
        return [
            dict(
                id=r["id"],
                state=r["state"],
                created=r["created"],
                mode=r["spec"]["mode"],
                sample_count=len(r["spec"]["samples"]),
                models=[p["name"] for p in r["spec"]["profiles"]],
            )
            for r in service.store.runs()
        ]

    @app.post("/api/runs", status_code=201)
    async def create_run(body: RunRequest):
        run = service.create_run(**body.model_dump())
        service.schedule(run["id"])
        return run

    @app.get("/api/runs/{rid}")
    async def run(rid: str):
        return service.store.run(rid)

    @app.post("/api/runs/{rid}/{action}")
    async def action(rid: str, action: str):
        if action == "stop":
            return service.stop(rid)
        if action not in ("resume", "retry"):
            raise ValueError("操作必须为 stop、resume 或 retry")
        service.schedule(rid, action)
        return service.store.run(rid)

    @app.get("/api/runs/{rid}/results")
    async def results(
        rid: str,
        offset: int = 0,
        limit: int = 50,
        history: bool = False,
        errors_only: bool = False,
    ):
        run = service.store.run(rid)
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("无效分页参数")
        all_rows = service.store.attempts(rid, latest=not history)
        if errors_only:
            all_rows = [
                r
                for r in all_rows
                if r["status"] != "success"
                or (
                    not r.get("correct")
                    if run["spec"]["mode"] == "eot"
                    else r.get("cer") not in (0, None)
                )
            ]
        rows = all_rows[offset : offset + limit]
        for row in rows if run["spec"]["mode"] != "eot" else []:
            row["diff"] = [
                {
                    "op": op,
                    "reference": row["reference_text"][a:b],
                    "hypothesis": row["text"][c:d],
                }
                for op, a, b, c, d in difflib.SequenceMatcher(
                    None, row["reference_text"], row["text"], autojunk=False
                ).get_opcodes()
            ]
        return {"total": len(all_rows), "rows": rows}

    @app.get("/api/runs/{rid}/report")
    async def report(rid: str):
        return service.report(rid)

    @app.get("/api/runs/{rid}/export/{format}")
    async def export(rid: str, format: str):
        report = service.report(rid)
        content, media = export_report(
            report,
            service.store.attempts(rid, latest=False),
            format,
            service.store.all_events(rid) if format == "json" else None,
        )
        return Response(
            content,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{rid}.{format}"'},
        )

    @app.get("/api/runs/{rid}/audio/{sid:path}")
    async def audio(rid: str, sid: str):
        run = service.store.run(rid)
        if run["spec"]["mode"] == "eot":
            raise ValueError("EOT 文本评测没有音频")
        samples = run["spec"]["samples"]
        sample = next((s for s in samples if s["id"] == sid), None)
        if sample is None:
            raise ValueError("样本不存在")
        return FileResponse(sample["audio_path"])

    @app.get("/api/runs/{rid}/events")
    async def events(rid: str, request: Request, after: int = 0):
        service.store.run(rid)
        try:
            cursor = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            raise ValueError("无效事件游标") from None

        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                batch = service.store.events(rid, cursor)
                for event in batch:
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                state = service.store.run(rid)["state"]
                if len(batch) < 500 and state not in ("queued", "running", "stopping"):
                    yield f"event: done\ndata: {json.dumps({'state': state})}\n\n"
                    return
                if len(batch) < 500:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
