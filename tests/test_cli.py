from pathlib import Path
import subprocess


def test_single_command_runs_and_writes(tmp_path: Path):
    out = tmp_path / "out.parquet"
    result = subprocess.run(
        ["xenmorph", str(tmp_path), "--out", str(out)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "run_metadata.json").exists()
    assert out.exists()
