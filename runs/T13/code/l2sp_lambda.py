"""T13 L2-SP 的 λ：λ·d² 占（第 2000 步附近的平均损失 + λ·d²）的 10%。d² 按 finetune/parts/l2sp.py 的定义，用 T13/main 的 ckpt_2000 算。"""
import json, sys
import torch

R = "/remote-home/xufang/YGC/wt/worktree-roma-batch2-122/runs/T13/ckpt/main"
base = torch.load("/remote-home/xufang/YGC/weights/anymatch/RoMa_AnyMatch.pth", map_location="cpu", weights_only=False)["model"]
ck = torch.load(f"{R}/ckpt_2000.pt", map_location="cpu", weights_only=False)
d2_dec = d2_vgg = 0.0
for k, v in ck["model"].items():
    if not v.is_floating_point() or "running_" in k:
        continue
    d = float(((v.float() - base[k].float()) ** 2).sum())
    if k.startswith("decoder."):
        d2_dec += d
    elif k.startswith("encoder.cnn"):
        d2_vgg += d
d2_lora = sum(float(((l["B"] @ l["A"]) * l["scale"]).pow(2).sum()) for l in ck["dinov2_lora"].values())
L = [json.loads(l) for l in open(f"{R}/log.jsonl")]
loss = sum(x["loss"] for x in L if 1900 <= x["step"] <= 2100) / sum(1 for x in L if 1900 <= x["step"] <= 2100)
d2 = d2_dec + d2_vgg + d2_lora
lam = loss / (9 * d2)
print(json.dumps({"d2_decoder": d2_dec, "d2_vgg": d2_vgg, "d2_lora": d2_lora, "d2": d2, "loss_1900_2100": loss,
                  "lambda": lam}, indent=1))
