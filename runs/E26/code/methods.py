"""E26 的方法：一个方法 = 一份存档（#206）。方法名 → (实验, 运行, 步)；zs 的步为 0（未训练，用 E19 gauss8 的配置）。"""
COLLAPSED = (("E15", "loo_t02"), ("E15", "none_t02"), ("E15", "none_t05"), ("E19", "clip"),
             ("E22", "f0002"), ("E23", "c16_s1"), ("E25", "f1_s1"))
HEALTHY = (("E19", "gauss8"), ("E23", "c16_s2"), ("E25", "fr1_s1"))
EVERY500 = (("E22", "f0002"), ("E23", "c16_s1"), ("E25", "f1_s1"))


def _methods():
    out = {}
    for e, r in COLLAPSED:
        for s in (range(500, 4001, 500) if (e, r) in EVERY500 else (1000, 4000)):
            out[f"{e.lower()}_{r}_{s}"] = (e, r, s)
    for e, r in HEALTHY:
        out[f"{e.lower()}_{r}_4000"] = (e, r, 4000)
    out["zs"] = ("E19", "gauss8", 0)
    return out


METHODS = _methods()
