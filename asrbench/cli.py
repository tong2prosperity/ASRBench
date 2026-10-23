"""Unified command line interface and compatibility entry point."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import load_config
from .service import BenchService
from .summary import export_report


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml")
    common.add_argument("--db", default="runtime/asrbench.sqlite3")
    p = argparse.ArgumentParser(description="ASRBench 客户数据集评测")
    commands = p.add_subparsers(dest="command", required=True)
    v = commands.add_parser("validate", parents=[common], help="导入并校验数据")
    v.add_argument("manifest", nargs="?")
    v.add_argument("--kind", choices=["asr", "eot"], default="asr")
    v.add_argument("--builtin-eot", action="store_true")
    v.add_argument("--audio-root")
    r = commands.add_parser("run", parents=[common], help="执行多模型评测")
    source = r.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--dataset-id")
    source.add_argument("--builtin-eot", action="store_true")
    r.add_argument("--audio-root")
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--mode", choices=["batch", "stream", "eot"], default="batch")
    r.add_argument("--concurrency", type=int)
    r.add_argument("--limit", type=int)
    r.add_argument("--prompt-file")
    r.add_argument("--system-prompt-file")
    r.add_argument("--label-policy", choices=["original", "stop_is_complete"])
    web = commands.add_parser("serve", parents=[common], help="启动本地网页")
    web.add_argument("--port", type=int, default=8765)
    for name in ("resume", "retry"):
        sub = commands.add_parser(name, parents=[common])
        sub.add_argument("run_id")
    report = commands.add_parser("report", parents=[common], help="导出任务报告")
    report.add_argument("run_id")
    report.add_argument("--format", choices=["md", "json", "csv"], default="md")
    report.add_argument("--output", required=True)
    return p


async def execute(args):
    service = BenchService(load_config(args.config), args.db)
    try:
        if args.command == "validate":
            if args.builtin_eot:
                if args.manifest:
                    raise ValueError("内置数据与外部清单不能同时指定")
                d = await service.import_builtin_eot()
            elif args.manifest:
                d = await service.import_dataset(
                    args.manifest, args.audio_root, args.kind
                )
            else:
                raise ValueError("请指定清单路径或 --builtin-eot")
            print(
                json.dumps(
                    {k: v for k, v in d.items() if k != "samples"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1 if d["errors"] else 0
        if args.command == "report":
            content, _ = export_report(
                service.report(args.run_id),
                service.store.attempts(args.run_id, latest=False),
                args.format,
                service.store.all_events(args.run_id)
                if args.format == "json"
                else None,
            )
            dest = Path(args.output).expanduser()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            print(f"已导出：{dest}")
            return 0
        if args.command == "run":
            did = args.dataset_id
            if args.builtin_eot and args.mode != "eot":
                raise ValueError("内置 EOT 数据请使用 --mode eot")
            if args.manifest or args.builtin_eot:
                dataset = (
                    await service.import_builtin_eot()
                    if args.builtin_eot
                    else await service.import_dataset(
                        args.manifest,
                        args.audio_root,
                        "eot" if args.mode == "eot" else "asr",
                    )
                )
                if dataset["errors"]:
                    print(json.dumps(dataset["errors"], ensure_ascii=False, indent=2))
                    return 1
                did = dataset["id"]
            run = service.create_run(
                did,
                args.models,
                args.mode,
                args.concurrency,
                args.limit,
                prompt_template=Path(args.prompt_file).read_text(encoding="utf-8")
                if args.prompt_file
                else None,
                system_prompt=Path(args.system_prompt_file).read_text(encoding="utf-8")
                if args.system_prompt_file
                else None,
                label_policy=args.label_policy,
            )
            rid = run["id"]
            task = service.schedule(rid)
        else:
            rid = args.run_id
            task = service.schedule(rid, args.command)
        print(f"任务：{rid}", flush=True)
        cursor = 0
        while not task.done():
            for event in service.store.events(rid, cursor):
                cursor = event["id"]
                if event["kind"] == "result":
                    print(
                        f"{event['profile_id']} / {event['sample_id']}: {event['status']}",
                        flush=True,
                    )
            await asyncio.sleep(0.2)
        await task
        report = service.report(rid)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k != "spec"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return (
            0
            if report["state"] == "completed"
            and all(p["success"] == p["total"] for p in report["profiles"])
            else 1
        )
    finally:
        await service.close()


async def legacy(argv):
    """Accept the original flags, but execute through the new service."""
    p = argparse.ArgumentParser(
        description="旧版参数兼容入口；新任务推荐使用 asrbench run"
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--provider")
    p.add_argument("--dataset", choices=["test", "dev", "both"])
    p.add_argument("--concurrency", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--output")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.resume:
        raise ValueError(
            "新存储按 run_id 恢复，请使用 asrbench resume RUN_ID；不会混入旧结果"
        )
    cfg = load_config(args.config)
    service = BenchService(cfg)
    try:
        bench = cfg.get("bench", {})
        root = Path(cfg["_path"]).parent / bench.get("corpus_dir", "corpus/zh-CN")
        split = args.dataset or bench.get("dataset", "test")
        from .dataset import load_dataset
        from .store import uid

        path = Path(service.store.path).parent / "imports" / (uid("legacy") + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "sample_id": s.id,
                        "audio_uri": str(Path(s.audio_path).resolve()),
                        "gold_text": s.ground_truth,
                    },
                    ensure_ascii=False,
                )
                for s in load_dataset(str(root), split)
            ),
            encoding="utf-8",
        )
        dataset = await service.import_dataset(str(path))
        models = [
            p["profile_id"]
            for p in cfg["models"]
            if p.get("enabled", True)
            and p["provider"] != "eot_chat"
            and (
                not args.provider
                or p["profile_id"] == args.provider
                or p["provider"] == args.provider
            )
        ]
        run = service.create_run(
            dataset["id"],
            models,
            concurrency=args.concurrency or bench.get("concurrency", 3),
            limit=args.limit,
        )
        print(f"任务：{run['id']}", flush=True)
        await service.schedule(run["id"])
        report = service.report(run["id"])
        target = Path(args.output or bench.get("output_dir", "results")) / run["id"]
        target.mkdir(parents=True, exist_ok=True)
        for fmt in ("json", "md"):
            text, _ = export_report(
                report,
                service.store.attempts(run["id"], latest=False),
                fmt,
                service.store.all_events(run["id"]) if fmt == "json" else None,
            )
            (target / f"report.{fmt}").write_text(text, encoding="utf-8")
        print(f"报告：{target}")
        return (
            0
            if report["state"] == "completed"
            and all(p["success"] == p["total"] for p in report["profiles"])
            else 1
        )
    finally:
        await service.close()


def main():
    argv = sys.argv[1:]
    try:
        if argv and argv[0].startswith("--") and argv[0] not in ("--help", "-h"):
            code = asyncio.run(legacy(argv))
        else:
            args = parser().parse_args(argv)
            if args.command == "serve":
                import uvicorn
                from .web import create_app

                app = create_app(load_config(args.config), args.db)
                uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
                code = 0
            else:
                code = asyncio.run(execute(args))
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("已中断；可使用任务 ID 恢复。", file=sys.stderr)
        code = 130
    raise SystemExit(code)
