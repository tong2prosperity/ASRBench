from fastapi.testclient import TestClient

from asrbench.service import BenchService
from asrbench.web import create_app
from test_service import FakeProvider


def test_web_import_create_results_report_events(tmp_path, manifest, config):
    s = BenchService(
        config, tmp_path / "bench.db", provider_factory=lambda *args: FakeProvider()
    )
    with TestClient(create_app(service=s)) as client:
        assert client.get("/").status_code == 200
        assert "test-only" not in client.get("/api/models").text
        d = client.post(
            "/api/datasets",
            json={"manifest_path": str(manifest[0]), "audio_root": str(manifest[1])},
        ).json()
        assert not d["errors"]
        response = client.post(
            "/api/runs", json={"dataset_id": d["id"], "profile_ids": ["alpha", "beta"]}
        )
        assert response.status_code == 201
        rid = response.json()["id"]
        with client.stream("GET", f"/api/runs/{rid}/events") as events:
            body = "".join(events.iter_text())
        assert "event: done" in body
        assert '"kind": "result"' in body
        report = client.get(f"/api/runs/{rid}/report").json()
        assert report["common_count"] == 2
        assert report["profiles"][0]["accuracy"]["cer"] == 0
        rows = client.get(f"/api/runs/{rid}/results?limit=1").json()
        assert len(rows["rows"]) == 1 and rows["total"] == 4
        assert rows["rows"][0]["diff"][0]["op"] == "equal"
        assert client.get(f"/api/runs/{rid}/audio/1").status_code == 200
        for fmt in ("json", "csv", "md"):
            r = client.get(f"/api/runs/{rid}/export/{fmt}")
            assert r.status_code == 200
            assert "attachment" in r.headers["content-disposition"]
        last = max(x["id"] for x in s.store.events(rid))
        body = client.get(
            f"/api/runs/{rid}/events", headers={"Last-Event-ID": str(last)}
        ).text
        assert '"kind": "result"' not in body
        assert "event: done" in body
        assert client.get(f"/api/runs/{rid}/results?limit=0").status_code == 400
        assert client.get(f"/api/runs/{rid}/audio/missing").status_code == 400
        assert client.post(f"/api/runs/{rid}/unknown", json={}).status_code == 400


def test_invalid_import_and_run(tmp_path, config):
    with TestClient(
        create_app(service=BenchService(config, tmp_path / "bench.db"))
    ) as c:
        assert c.post("/api/datasets", json={}).status_code == 400
        assert (
            c.post(
                "/api/runs", json={"dataset_id": "missing", "profile_ids": []}
            ).status_code
            == 400
        )
        assert c.post("/api/datasets", json={"manifest_text": "invalid json"}).json()[
            "errors"
        ]


def test_large_event_replay_no_loss_and_json_audit(tmp_path, manifest, config):
    import asyncio

    s = BenchService(config, tmp_path / "bench.db")
    d = asyncio.run(s.import_dataset(*manifest))
    rid = s.create_run(d["id"], ["alpha"])["id"]
    s.store.state(rid, "completed")
    for i in range(510):
        s.store.event(rid, "synthetic", {"sequence": i})
    # Startup correctly leaves completed work untouched.
    with TestClient(create_app(service=s)) as c:
        body = c.get(f"/api/runs/{rid}/events").text
        assert body.count('"kind": "synthetic"') == 510
        report = c.get(f"/api/runs/{rid}/export/json").json()
        assert sum(e["kind"] == "synthetic" for e in report["events"]) == 510
