"""Locate the official weight, final weight, and intermediate Val predictions."""

from workbench.records import load_runs


def prediction_location(run_dir, update, last):
    if update == 0:
        return run_dir, "init"
    if update == last:
        return run_dir, "main"
    return run_dir / "sweep/main/S", f"step{update}"


def run_record(run_dir):
    return load_runs(run_dir.parent)[run_dir.name]
