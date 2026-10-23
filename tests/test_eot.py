import asyncio
import json
from collections import Counter

import httpx
import pytest
from fastapi.testclient import TestClient

from asrbench import cli, eot
from asrbench.service import BenchService
from asrbench.summary import export_report
from asrbench.web import create_app


@pytest.fixture
def eot_config():
    return {
        "models": [
            {
                "profile_id": name,
                "name": name,
                "provider": "eot_chat",
                "enabled": True,
                "params": {
                    "base_url": "http://eot.test/v1",
                    "model": name,
                    "api_key": "test-only-secret",
                    "timeout": 1,
                },
            }
            for name in ("eot_a", "eot_b")
        ]
    }


@pytest.fixture
def eot_manifest(tmp_path):
    path = tmp_path / "eot.jsonl"
    rows = [
        {"sample_id": "done", "text": "谢谢", "label": 1, "category": "conversation"},
        {
            "sample_id": "unfinished",
            "text": "因为",
            "label": 0,
            "category": "conversation",
        },
        {
            "sample_id": "wait",
            "text": "先别说话",
            "label": 0,
            "category": "silence_request",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return path


def factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        (" 1\n", 1),
        ("10", None),
        ("1 or 0", None),
        ("判断为1", None),
        ("", None),
        ("true", None),
        (None, None),
    ],
)
def test_strict_binary_parser(text, expected):
    assert eot.parse_output(text) == expected


def test_bundled_data_only_annotations_and_original_labels():
    path = eot.RESOURCE_DIR / "base.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows) == 1134
    assert all(set(r) == {"text", "label", "category"} for r in rows)
    assert Counter(r["label"] for r in rows) == {1: 508, 0: 626}
    assert sum(r["category"] == "silence_request" for r in rows) == 200
    result = eot.validate_manifest(path)
    assert not result["errors"] and result["count"] == 1134
    assert len({r["id"] for r in result["samples"]}) == 1134
    assert eot.validate_prompt(eot.defaults()["prompt_template"])


def test_invalid_manifest(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        "\n".join(
            [
                "null",
                "[]",
                json.dumps({"text": "x", "label": True}),
                json.dumps({"text": "x", "label": 2}),
                json.dumps({"text": "", "label": 0}),
                json.dumps({"id": "duplicate", "text": "x", "label": 0}),
                json.dumps({"id": "duplicate", "text": "x", "label": 0}),
            ]
        )
    )
    assert len(eot.validate_manifest(p)["errors"]) == 6
    p.write_text("")
    assert eot.validate_manifest(p)["errors"]


def test_metrics_confusion_and_undefined_denominators():
    rows = [
        {"status": "success", "label": label, "pred": pred}
        for label, pred in [(1, 1), (1, 0), (0, 1), (0, 0)]
    ]
    rows.append({"status": "invalid", "label": 1, "pred": None})
    m = eot.classification_metrics(rows)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["precision"] == m["recall"] == m["fpr"] == 0.5
    assert eot.classification_metrics([])["accuracy"] is None
    assert eot.classification_metrics(rows[3:4])["recall"] is None


def test_chat_payload_prompt_label_policy_snapshot_and_exports(
    tmp_path, eot_config, eot_manifest
):
    requests = []

    async def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-only-secret"
        assert (
            payload["max_tokens"] == 4
            and payload["temperature"] == 0
            and payload["stream"] is False
        )
        assert payload["messages"][0] == {"role": "system", "content": "System fixture"}
        text = payload["messages"][-1]["content"]
        pred = "0" if "因为" in text else "1"
        return httpx.Response(200, json={"choices": [{"message": {"content": pred}}]})

    async def scenario():
        s = BenchService(
            eot_config, tmp_path / "bench.db", eot_client_factory=factory(handler)
        )
        d = await s.import_dataset(eot_manifest, kind="eot")
        r = s.create_run(
            d["id"],
            ["eot_a", "eot_b"],
            mode="eot",
            prompt_template="Input: {text}",
            system_prompt="System fixture",
            label_policy="stop_is_complete",
        )
        eot_config["eot"] = {"prompt_template": "CHANGED %s"}
        await s.schedule(r["id"])
        report = s.report(r["id"])
        p = report["profiles"][0]
        assert report["common_count"] == 3 and report["mode"] == "eot"
        assert p["metrics"]["accuracy"] == 1 and p["core_metrics"]["count"] == 2
        wait = next(a for a in s.store.attempts(r["id"]) if a["sample_id"] == "wait")
        assert wait["source_label"] == 0 and wait["label"] == 1
        assert all(q["messages"][-1]["content"].startswith("Input: ") for q in requests)
        assert "test-only-secret" not in json.dumps(report)
        for format in ("md", "csv", "json"):
            output, _ = export_report(report, s.store.attempts(r["id"]), format)
            assert output and "test-only-secret" not in output
            if format == "csv":
                assert "source_label,label,pred" in output
            if format == "md":
                assert "EOT" in output and "CER" not in output
                assert output.index("|eot_b|") < output.index("## eot_a 分类明细")
        await s.close()

    asyncio.run(scenario())


def test_invalid_output_http_failure_retry_and_common_coverage(
    tmp_path, eot_config, eot_manifest
):
    retry = [False]

    async def handler(request):
        payload = json.loads(request.content)
        text = payload["messages"][-1]["content"]
        if payload["model"] == "eot_b" and not retry[0]:
            if "谢谢" in text:
                return httpx.Response(403, json={"error": "test-only-secret"})
            if "因为" in text:
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": "1 or 0"}}]}
                )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "1" if "谢谢" in text else "0"}}]
            },
        )

    async def scenario():
        s = BenchService(
            eot_config, tmp_path / "bench.db", eot_client_factory=factory(handler)
        )
        d = await s.import_dataset(eot_manifest, kind="eot")
        rid = s.create_run(
            d["id"], ["eot_a", "eot_b"], mode="eot", prompt_template="Input:%s"
        )["id"]
        await s.schedule(rid)
        report = s.report(rid)
        b = report["profiles"][1]
        assert b["request_failures"] == 1 and b["unparsed"] == 1 and b["success"] == 1
        assert report["common_count"] == 1
        assert "test-only-secret" not in str(s.store.attempts(rid))
        await s.schedule(rid, "resume")
        assert len(s.store.attempts(rid, latest=False)) == 6
        retry[0] = True
        await s.schedule(rid, "retry")
        assert len(s.store.attempts(rid, latest=False)) == 8
        assert s.report(rid)["common_count"] == 3
        await s.close()

    asyncio.run(scenario())


def test_timeout_and_stop_resume(tmp_path, eot_config, eot_manifest):
    started = asyncio.Event()
    slow = [True]

    async def handler(request):
        started.set()
        if slow[0]:
            await asyncio.Event().wait()
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    async def scenario():
        eot_config["models"][0]["params"]["timeout"] = 0.03
        s = BenchService(
            eot_config, tmp_path / "bench.db", eot_client_factory=factory(handler)
        )
        d = await s.import_dataset(eot_manifest, kind="eot")
        rid = s.create_run(d["id"], ["eot_a"], mode="eot", concurrency=1)["id"]
        await s.schedule(rid)
        assert all(r["status"] == "timeout" for r in s.store.attempts(rid))
        rid2 = s.create_run(d["id"], ["eot_b"], mode="eot", concurrency=1)["id"]
        started.clear()
        task = s.schedule(rid2)
        await started.wait()
        s.stop(rid2)
        await task
        assert s.store.run(rid2)["state"] == "cancelled"
        slow[0] = False
        await s.schedule(rid2, "resume")
        assert s.report(rid2)["profiles"][0]["success"] == 3
        await s.close()

    asyncio.run(scenario())


def test_eot_api_builtin_error_filter_and_wrong_kind(
    tmp_path, eot_config, eot_manifest, manifest
):
    async def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    s = BenchService(
        eot_config, tmp_path / "bench.db", eot_client_factory=factory(handler)
    )
    with TestClient(create_app(service=s)) as c:
        assert c.get("/api/models").json()[0]["modes"] == ["eot"]
        assert c.get("/api/eot/defaults").json()["builtin_count"] == 1134
        built = c.post("/api/datasets/builtin-eot", json={}).json()
        assert built["count"] == 1134 and not built["errors"] and built["kind"] == "eot"
        d = c.post(
            "/api/datasets", json={"manifest_path": str(eot_manifest), "kind": "eot"}
        ).json()
        r = c.post(
            "/api/runs",
            json={"dataset_id": d["id"], "profile_ids": ["eot_a"], "mode": "eot"},
        )
        assert r.status_code == 201
        rid = r.json()["id"]
        assert "event: done" in c.get(f"/api/runs/{rid}/events").text
        rows = c.get(f"/api/runs/{rid}/results?errors_only=true").json()
        assert rows["total"] == 2 and all(not a["correct"] for a in rows["rows"])
        assert c.get(f"/api/runs/{rid}/audio/done").status_code == 400
        assert c.get(f"/api/runs/{rid}/export/md").status_code == 200
        assert c.get(f"/api/runs/{rid}/export/json").json()["report"]["mode"] == "eot"
        asr = c.post(
            "/api/datasets",
            json={"manifest_path": str(manifest[0]), "audio_root": str(manifest[1])},
        ).json()
        assert (
            c.post(
                "/api/runs",
                json={"dataset_id": asr["id"], "profile_ids": ["eot_a"], "mode": "eot"},
            ).status_code
            == 400
        )
        assert (
            c.post(
                "/api/runs",
                json={"dataset_id": d["id"], "profile_ids": ["eot_a"], "mode": "batch"},
            ).status_code
            == 400
        )


def test_cli_builtin_and_eot_run(tmp_path, eot_config, monkeypatch, capsys):
    async def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    monkeypatch.setattr(cli, "load_config", lambda _: eot_config)
    monkeypatch.setattr(
        cli,
        "BenchService",
        lambda cfg, db: BenchService(cfg, db, eot_client_factory=factory(handler)),
    )
    common = ["--config", "unused", "--db", str(tmp_path / "cli.db")]
    assert (
        asyncio.run(
            cli.execute(cli.parser().parse_args(["validate", "--builtin-eot", *common]))
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["count"] == 1134
    args = cli.parser().parse_args(
        [
            "run",
            "--builtin-eot",
            "--mode",
            "eot",
            "--models",
            "eot_a",
            "--limit",
            "3",
            *common,
        ]
    )
    assert asyncio.run(cli.execute(args)) == 0
    text = capsys.readouterr().out
    assert '"mode": "eot"' in text


@pytest.mark.parametrize(
    "params",
    [
        {"base_url": "ftp://host", "model": "a"},
        {"base_url": "http://u:p@host/v1", "model": "a"},
        {"base_url": "http://host/v1", "model": "a", "extra_body": {"messages": []}},
        {
            "base_url": "http://host/v1",
            "model": "a",
            "prompt_template": "no placeholder",
        },
    ],
)
def test_bad_model_parameters(params):
    with pytest.raises(ValueError):
        eot.validate_params(params)
