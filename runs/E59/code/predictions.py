"""Locate the official weight, final weight, and intermediate Val predictions."""


def prediction_location(run_dir, update, last):
    if update == 0:
        return run_dir, "init"
    if update == last:
        return run_dir, "main"
    return run_dir / "sweep/main/S", f"step{update}"
