from __future__ import annotations
import json
from pathlib import Path
import typer

from .io import validate_input_dir
from .metrics import compute_morphometrics_stub


def main(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to Xenium export directory",
    ),
    out: Path = typer.Option("morphometrics.parquet", help="Output table path"),
    n_jobs: int = typer.Option(1, help="Parallel workers"),
):
    """
    Compute per-cell/nuclear morphometrics from Xenium export.
    Current implementation is a stub: validates input and writes an empty table
    with the expected schema, plus run_metadata.json.
    """
    input_dir = validate_input_dir(input_dir)
    df = compute_morphometrics_stub()  # TODO: replace with real pipeline

    out.parent.mkdir(parents=True, exist_ok=True)
    # write parquet (empty is fine; schema is useful for downstream testing)
    df.to_parquet(out)

    run_meta = {
        "ok": True,
        "input_dir": str(input_dir),
        "out": str(out.resolve()),
        "n_jobs": n_jobs,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "version": "0.1.0",
    }
    (out.parent / "run_metadata.json").write_text(json.dumps(run_meta, indent=2))
    typer.echo(f"[xenmorph] wrote {out} ({df.shape[0]} rows) and run_metadata.json")


def app():
    typer.run(main)


if __name__ == "__main__":
    app()
