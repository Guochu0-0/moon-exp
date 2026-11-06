"""E43 停止规则（#250）：训完第 1 遍训练集（第 250、500、…、2000 次更新共 8 个存档）时，8 个存档的 Val AUC@5 都低于
E41 `main` 同位置存档，或 8 个存档平均低于 E41 0.01 以上，就停训。在 gpfs worktree 里、训练进行中后台运行（只用 CPU）：

    nohup /opt/envs/wb/bin/python runs/E43/code/stop_rule.py > runs/E43/ckpt/stop_rule.log 2>&1 < /dev/null &

每 60 秒看一次：8 个存档的 Val 都评完（eval_queue 写的 .evalclaims/main_<step>/done）后，按逐对误差算 AUC@5
（workbench.evaluate.pair_error，失败记 ∞，同 workbench eval），与 E41 `main` 的 sweep/main/S/metrics.json 同位置比较，
写 extra/data/stop_rule.json（各存档两边的 AUC@5、差、评测机器，平均差，是否停）。要停时再建 ckpt/main/stop.request，
由 train_queue.py 结束训练。只判断一次，判断完退出。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
REPO = R.parents[1]
sys.path.insert(0, str(REPO))
from workbench.dataset import Dataset  # noqa: E402
from workbench.evaluate import pair_error  # noqa: E402
from workbench.protocol import auc  # noqa: E402

Y = Path("/remote-home/xufang/YGC")
E41 = Y / "moon-exp" / "runs" / "E41" / "sweep" / "main" / "S"
DATA = Y / "dataset" / "Moon"
STEPS = list(range(250, 2001, 250))
M = "main"


def errors(ds, st):
    p = R / "sweep" / M / "S" / "preds" / f"step{st}" / "val.jsonl"
    preds = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["pair"]] = r.get("A")
    e = [pair_error(ds, "val", q, preds.get(q)) for q in ds.labelled("val")]
    return np.array([np.inf if x is None else x for x in e], float)


def host(d, st):
    f = d / "preds" / f"step{st}" / "val.meta.json"
    return json.loads(f.read_text(encoding="utf-8")).get("host") if f.exists() else None


def main():
    claims = R / ".evalclaims"
    while not all((claims / f"{M}_{st}" / "done").exists() for st in STEPS):
        if (R / "launch" / f"{M}.json").exists() and json.loads((R / "launch" / f"{M}.json").read_text())[-1].get("status"):
            print("训练已结束，没等到 8 个存档", flush=True)
            return
        time.sleep(60)
    ds = Dataset(DATA)
    ref = json.loads((E41 / "metrics.json").read_text(encoding="utf-8"))["methods"]
    rows = []
    for st in STEPS:
        a = auc(errors(ds, st), 5.0)
        b = ref[f"step{st}"]["val"]["summary"]["auc@5"]
        rows.append({"update": st, "auc5": round(a, 4), "auc5_e41": round(b, 4), "diff": round(a - b, 4),
                     "host": host(R / "sweep" / M / "S", st), "host_e41": host(E41, st)})
    mean = float(np.mean([r["diff"] for r in rows]))
    all_below = all(r["diff"] < 0 for r in rows)
    stop = all_below or mean < -0.01
    out = {"rows": rows, "mean_diff": round(mean, 4), "all_below": all_below, "stop": stop,
           "rule": "8 个存档 AUC@5 都低于 E41 main 同位置存档，或平均低 0.01 以上则停"}
    (R / "extra" / "data").mkdir(parents=True, exist_ok=True)
    (R / "extra" / "data" / "stop_rule.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                                         encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)
    if stop:
        (R / "ckpt" / M / "stop.request").touch()


if __name__ == "__main__":
    main()
