from __future__ import annotations
from pathlib import Path


def validate_input_dir(input_dir: Path) -> Path:
    """Basic checks that the input path exists and looks like a Xenium export."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    # Later: assert presence of expected files (e.g., cells.csv.gz, nuclei.csv.gz, masks, metadata)
    return input_dir.resolve()
