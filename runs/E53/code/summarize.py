"""核对 #284 的 Test 图对与末次存档，并写论文绘图数据。

用法（服务器实验 worktree）：python runs/E53/code/summarize.py <gpfs 主 checkout>
"""
import argparse
import csv
import json
from pathlib import Path


E53 = Path(__file__).resolve().parents[1]
LEVELS = (16, 8, 4, 2, 1)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pairs(run, method):
    path = run / "preds" / method / "test.jsonl"
    ids = [json.loads(line)["pair"] for line in path.read_text(encoding="utf-8").splitlines()]
    if len(ids) != 953 or len(set(ids)) != len(ids):
        raise ValueError(f"{path} 需要 953 个互异 Test 图对，得到 {len(ids)} 行、{len(set(ids))} 个 ID")
    return set(ids)


def result(run, method, k, step, checkpoint, expected_pairs):
    ids = pairs(run, method)
    if ids != expected_pairs:
        raise ValueError(f"{run.name}/{method} 与 E41 的 Test 图对不同，缺少 {sorted(expected_pairs - ids)}，额外 {sorted(ids - expected_pairs)}")
    test = read_json(run / "metrics.json")["methods"][method]["test"]
    if test["summary"]["n"] != 953 or test["coverage"]["missing"]:
        raise ValueError(f"{run.name}/{method} Test 指标未覆盖全部 953 个图对")
    records = read_json(run / "launch" / f"{method}.json")
    eval_path = run / "launch" / f"{method}_eval.json"
    evaluation = read_json(eval_path)[-1] if eval_path.is_file() else records[-1]
    if records[-1]["status"] != "ok" or evaluation["status"] != "ok" or len(records) < (2 if k == 5 else 1):
        raise ValueError(f"{run.name}/{method} 缺少成功的启动记录")
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(checkpoint)
    training = records[0] if k == 5 else read_json(checkpoint.with_name("args.json"))
    return {
        "k": k,
        "position_reward_scales": " ".join(map(str, LEVELS[:k])),
        "end_step": step,
        "test_auc5_percent": 100 * test["summary"]["auc@5"],
        "checkpoint": f"runs/{run.name}/ckpt/{method}/{checkpoint.name}",
        "config_file": records[0 if k == 5 else -1]["config_file"],
        "training_commit": training["commit"],
        "test_commit": evaluation["commit"],
        "test_prediction": f"runs/{run.name}/preds/{method}/test.jsonl",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="有 E41 末次权重的 gpfs 主 checkout 路径")
    args = parser.parse_args()
    reference = E53.parent / "E41"
    reference_method = "main__screen0.8"
    expected_pairs = pairs(reference, reference_method)
    rows = []
    for k in range(5):
        method = f"k{k}"
        ckpt = E53 / "ckpt" / method
        log = ckpt / "log.jsonl"
        last_step = None
        with log.open(encoding="utf-8") as f:
            for line in f:
                last_step = json.loads(line)["step"]
        if last_step is None:
            raise ValueError(f"{log} 没有训练步骤")
        selected = read_json(E53 / "sweep" / method / "peak.json")
        if selected["step"] != last_step:
            raise ValueError(f"{method} 的终评权重在第 {selected['step']} 步，训练结束于第 {last_step} 步")
        rows.append(result(E53, method, k, last_step, ckpt / f"ckpt_{last_step}.pt", expected_pairs))

    checkpoint = args.reference / "runs" / "E41" / "ckpt" / reference_method / "ckpt_7907.pt"
    rows.append(result(reference, reference_method, 5, 7907, checkpoint, expected_pairs))
    for i, row in enumerate(rows):
        auc = row["test_auc5_percent"]
        row["delta_pp"] = auc - rows[i - 1]["test_auc5_percent"] if i else ""

    out = E53 / "extra" / "data" / "test_auc5.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
