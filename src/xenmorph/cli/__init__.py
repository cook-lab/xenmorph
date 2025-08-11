# =============================
# xenmorph/cli/__init__.py (clean root CLI, no legacy shim)
# ============================= (clean root CLI, no legacy shim)
# =============================
from __future__ import annotations
import typer

from .metrics_compute import app as metrics_app

# Root Typer app — subcommands live under "metrics"
app = typer.Typer(no_args_is_help=True, help="XenMorph CLI")
app.add_typer(metrics_app, name="metrics")

# Expose for entry point
main_app = app


def main():
    app(prog_name="xenmorph")
