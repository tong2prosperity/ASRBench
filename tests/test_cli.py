import asyncio
import json
import shutil

from asrbench import cli
from asrbench.service import BenchService
from test_service import FakeProvider


def test_cli_validate_run_and_report(tmp_path, manifest, config, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "BenchService",
        lambda cfg, db: BenchService(
            cfg, db, provider_factory=lambda *a: FakeProvider()
        ),
    )
    common = ["--config", "unused.yaml", "--db", str(tmp_path / "cli.db")]
    args = cli.parser().parse_args(
        ["validate", str(manifest[0]), "--audio-root", str(manifest[1]), *common]
    )
    assert asyncio.run(cli.execute(args)) == 0
    output = json.loads(capsys.readouterr().out)
    args = cli.parser().parse_args(
        ["run", "--dataset-id", output["id"], "--models", "alpha", "beta", *common]
    )
    assert asyncio.run(cli.execute(args)) == 0
    text = capsys.readouterr().out
    rid = text.splitlines()[0].split("：")[1]
    args = cli.parser().parse_args(
        [
            "report",
            rid,
            "--output",
            str(tmp_path / "report.json"),
            "--format",
            "json",
            *common,
        ]
    )
    assert asyncio.run(cli.execute(args)) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["report"]["common_count"] == 2
    assert report["events"]


def test_legacy_flags_use_new_pipeline(tmp_path, manifest, config, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "clips").mkdir(parents=True)
    shutil.copyfile(manifest[1] / "1.wav", corpus / "clips" / "1.wav")
    (corpus / "test.tsv").write_text("path\tsentence\n1.wav\t你好 world\n")
    config.update(
        _path=str(tmp_path / "config.yaml"),
        bench={"corpus_dir": "corpus", "output_dir": str(tmp_path / "outputs")},
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "BenchService",
        lambda cfg: BenchService(
            cfg, tmp_path / "cli.db", provider_factory=lambda *a: FakeProvider()
        ),
    )
    assert asyncio.run(cli.legacy(["--provider", "alpha"])) == 0
    report = next((tmp_path / "outputs").glob("*/report.json"))
    assert (
        json.loads(report.read_text())["report"]["profiles"][0]["accuracy"]["cer"] == 0
    )
