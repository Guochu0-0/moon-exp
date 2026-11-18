"""E50 续评与最终验收共用的逐存档完整性检查，不重算指标。"""
import json
import zipfile
from pathlib import Path


def incomplete_reason(run, step, split, expected):
    expected = set(expected)
    raw = Path(run) / f"sweep/main/match/step{step}"
    pred = Path(run) / f"sweep/main/S/preds/step{step}"
    try:
        rows = {}
        for name, directory in (("raw", raw), ("pred", pred)):
            items = [json.loads(s) for s in (directory / f"{split}.jsonl").read_text().splitlines() if s.strip()]
            if len(items) != len(expected) or {r["pair"] for r in items} != expected:
                return f"{name}: incomplete or duplicate pair coverage"
            rows[name] = items
        meta = json.loads((raw / f"{split}.meta.json").read_text())
        if not meta.get("finished") or meta.get("n_pairs") != len(expected):
            return "raw: missing completion metadata"
        meta = json.loads((pred / f"{split}.meta.json").read_text())
        if not meta.get("written"):
            return "pred: missing completion metadata"
        # 真正的逐对推理错误保留为失败；正常返回的点对必须已经写入 NPZ。
        required = {r["pair"].replace("/", "__") + ".npy" for r in rows["raw"] if "error" not in r}
        for path in (raw / f"{split}.npz", pred / f"{split}_matches.npz"):
            if not required and not path.exists():
                continue
            with zipfile.ZipFile(path) as z:
                if not required <= set(z.namelist()):
                    return f"{path.name}: unflushed matches"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        return f"{type(e).__name__}: {e}"
    return None
