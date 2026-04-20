import json
import os
from rich.console import Console
from rich.table import Table


def save_results(results: dict, output_dir: str, provider_name: str):
    """Save per-sample results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{provider_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return path


def print_summary(all_summaries: list[dict]):
    """Print a rich table summarizing all providers.

    Shows corpus-level CER/WER (standard academic aggregate, weighted by
    reference length) as the primary metric, with per-sample arithmetic
    mean in parentheses for comparison.
    """
    console = Console()
    table = Table(title="ASR Benchmark Results (corpus-level; mean in parens)")
    table.add_column("Provider", style="cyan")
    table.add_column("CER %", justify="right")
    table.add_column("WER %", justify="right")
    table.add_column("Avg RTF", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Errors", justify="right", style="red")

    # Sort best-to-worst by corpus CER for easy comparison.
    ranked = sorted(all_summaries, key=lambda s: s.get("cer", 1.0))
    for s in ranked:
        cer = s.get("cer", s.get("avg_cer", 0))
        wer = s.get("wer", s.get("avg_wer", 0))
        mean_cer = s.get("mean_cer", cer)
        mean_wer = s.get("mean_wer", wer)
        table.add_row(
            s["provider"],
            f"{cer * 100:.2f} ({mean_cer * 100:.2f})",
            f"{wer * 100:.2f} ({mean_wer * 100:.2f})",
            f"{s['avg_rtf']:.3f}",
            str(s["total"]),
            str(s["errors"]),
        )
    console.print(table)
