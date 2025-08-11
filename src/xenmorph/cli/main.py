import typer
from .metrics_compute import app as metrics_app

main_app = typer.Typer()
main_app.add_typer(metrics_app, name="metrics")


def main():
    main_app()
