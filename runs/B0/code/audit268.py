"""Preserve recorded provenance and distinguish code checks from raw git status."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[3]
run = root / "runs/B0"


def git(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


launches = {method: json.loads((run / "launch" / (method + ".json")).read_text())
            for method in ("splg", "roma")}
for records in launches.values():
    assert all(row["end"] and row["status"] == "ok" and not row["dirty"] and not row["allow_dirty"] for row in records)
recorded = {method: [{key: row.get(key) for key in ("start", "end", "status", "commit", "dirty", "allow_dirty", "args", "cmd")}
                     for row in records] for method, records in launches.items()}
commits = sorted({row["commit"] for records in launches.values() for row in records})
scope = ["baselines", "configs", "moonlib", "workbench", "third_party", ".gitmodules", "runs/B0/code/supplement268.py"]
diffs = {commit: git("diff", commit, "HEAD", "--", *scope) for commit in commits}
assert all(not value for value in diffs.values())
source_commit = launches["roma"][-1]["commit"]
source = git("show", source_commit + ":baselines/match.py")
git_head = source[source.index("def git_head("):source.index("\n\ndef env_info(")]
pred_meta, raw_meta = {}, {}
for method in ("splg", "roma"):
    for split in ("val", "test"):
        key = method + "/" + split
        pred_meta[key] = json.loads((run / "preds" / method / (split + ".meta.json")).read_text())
        assert not pred_meta[key]["dirty"]
        raw = json.loads((run / "extra/supplement268" / (method + "_" + split + "_meta.json")).read_text())
        raw_meta[key] = {name: raw.get(name) for name in ("commit", "repo_commit", "started", "finished", "n_pairs", "n_errors_this_session")}
prior = json.loads(git("show", source_commit + ":runs/B0/launch/roma.json"))
logs = []
for name in ("selfchecks", "splg_val", "splg_test", "roma_val", "roma_test"):
    path = run / "ckpt" / ("supplement268_" + name + ".log")
    logs.append(dict(path=str(path.relative_to(run)), bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
output = dict(recorded_launch_fields=recorded, recorded_prediction_meta=pred_meta, recorded_raw_meta=raw_meta,
              inference_code_diff_scope=scope, inference_code_diff_from_recorded_commits=diffs,
              generic_git_head_definition=dict(commit=source_commit, path="baselines/match.py", source=git_head),
              code_check_definition=dict(commit=source_commit, launch="workbench/launch.py:repo_state; _CODE excludes runs, _RUN_CODE includes run code/configs",
                                         predictions="workbench/records.py:git_state; excludes runs"),
              known_tracked_runtime_record=dict(path="runs/B0/launch/roma.json", tracked_at_source=True,
                                               entries_at_source=len(prior), entries_now=len(launches["roma"])),
              original_per_file_status=None,
              limitation="Raw git_head did not save its original per-file status listing. No historical listing is reconstructed. The recorded code checks and differing status scopes can explain the raw dirty suffix.",
              contemporaneous_stdout_logs=logs)
path = run / "extra/supplement268/provenance_audit.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("provenance: 6 clean completed launches; 4 clean prediction metas; inference code/config diffs empty; original per-file status unavailable")
