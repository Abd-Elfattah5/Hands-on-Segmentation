"""Typer CLI `seg` (skeleton). Commands are added during implementation; see HANDOFF.md §6."""

from typing import Optional

import typer

__version__ = "0.1.0"

app = typer.Typer(
    name="seg",
    help="Clothes segmentation (U-Net, BCE + Dice) CLI for assignment CVCY002.",
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Hands-on Segmentation version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Hands-on Segmentation command-line interface."""


if __name__ == "__main__":
    app()
