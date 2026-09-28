"""画布存储：runs/canvas.json 的读写，以及没有坐标的实验的默认位置。

    {
      "experiments": {"B0": {"x": 120, "y": 80}},
      "groups":      [{"id": "g1", "x": -20, "y": -20, "w": 800, "h": 260, "title": "zero-shot", "color": "c1"}],
      "stickies":    [{"id": "s1", "x": 0, "y": 300, "w": 220, "h": 120, "text": "..."}]
    }

坐标是世界坐标（px），原点在左上。实验卡片宽度固定，只存 x、y。视口存在浏览器 localStorage，不在这里。
"""
from __future__ import annotations

import json
from pathlib import Path

NODE_W = 300        # 实验卡片宽度，与前端 .node 一致
GAP_X = 120         # 父子之间的水平间距
GAP_Y = 40          # 上下相邻卡片的最小间距
STEP_Y = 30         # 找空位时每次往下挪多少
SHOWN_METHODS = 4   # 卡片上默认显示的方法行数，与前端一致


def path(runs_dir: Path) -> Path:
    return Path(runs_dir) / "canvas.json"


def load(runs_dir: Path) -> dict:
    p = path(runs_dir)
    c = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {"experiments": dict(c.get("experiments", {})), "groups": list(c.get("groups", [])),
            "stickies": list(c.get("stickies", []))}


def save(runs_dir: Path, canvas: dict, ids) -> dict:
    """写 canvas.json：丢掉目录已不存在的实验，key 排序，缩进 2 格。坐标取整。返回写入的内容。"""
    ids = set(ids)
    out = {
        "experiments": {k: {"x": round(v["x"]), "y": round(v["y"])}
                        for k, v in canvas.get("experiments", {}).items() if k in ids},
        "groups": list(canvas.get("groups", [])),
        "stickies": list(canvas.get("stickies", [])),
    }
    text = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path(runs_dir).write_text(text, encoding="utf-8", newline="\n")
    return out


def node_height(n_methods: int, n_shown: int, lit: bool) -> int:
    """卡片高度的估计，只用于摆放时避让，与前端 CSS 大致一致。"""
    if not lit:
        return 124
    rows = max(1, n_shown)
    return 34 + 29 + rows * 26 + (24 if n_methods > rows else 8)


def place(nodes: dict[str, dict], positions: dict[str, dict]) -> dict[str, dict]:
    """给没有坐标的实验算默认位置，返回全部实验的坐标。

    nodes: {id: {"parent": str | None, "h": 估计高度}}；positions: 已有坐标 {id: {"x", "y"}}。
    有父实验的放在父实验右侧，与父实验顶端对齐；兄弟实验和其他挡路的实验往下避让。
    根实验放在全部已有实验的下方。按 id 顺序处理，父实验总是先于子实验落位。
    """
    out = {k: {"x": v["x"], "y": v["y"]} for k, v in positions.items() if k in nodes}
    boxes = [(v["x"], v["y"], nodes[k]["h"]) for k, v in out.items()]

    def free(x, y, h):
        while any(x < bx + NODE_W + GAP_Y and x + NODE_W + GAP_Y > bx and y < by + bh + GAP_Y and y + h + GAP_Y > by
                  for bx, by, bh in boxes):
            y += STEP_Y
        return y

    def put(k, seen=()):
        if k in out:
            return out[k]
        p = nodes[k]["parent"]
        if p in nodes and p not in seen:
            pp = put(p, (*seen, k))
            x, y = pp["x"] + NODE_W + GAP_X, pp["y"]
        else:
            x, y = 0, max((by + bh + GAP_Y * 2 for _, by, bh in boxes), default=0)
        y = free(x, y, nodes[k]["h"])
        out[k] = {"x": x, "y": y}
        boxes.append((x, y, nodes[k]["h"]))
        return out[k]

    for k in sorted(nodes):
        put(k)
    return out
