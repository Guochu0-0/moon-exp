"""RMSO-ConvNeXt（yeyuanxin110/RMSO-ConvNeXt，TGRS 2025，third_party/RMSO-ConvNeXt，19c8faa）。

它是**模板匹配**网络：伪孪生全分辨率特征（8 通道、逐像素 L2 归一化，model/RMSO_ConvNeXt.py:176-203），
`fft_match_batch` 做 FFT 加速的 SSD（utils/match_template_function.py:7-59），argmin 得 SAR 模板左上角在光学参考图中的
整数偏移（demo.py:67）。官方尺寸：光学参考 512²、SAR 模板 256²。官方只给单模板平移，**没有多点对应流程**。

这里的做法（我们的设计，非官方）：保持官方模板 256² 与整张 512 参考图的搜索方式，在 SAR 上按 stride 取模板网格
（stride 32 → 9×9 = 81 个），每个模板给一对对应点 = 模板中心：kp1 = (sx+(T−1)/2, sy+(T−1)/2)，
kp0 = (x+(T−1)/2, y+(T−1)/2)，(x, y) 为 SSD argmin。之后走统一的仿射 RANSAC。conf = −min SSD。
- 输入：(B,1,H,W) float [0,1]；模型对每张输入（每个模板）各自 z-score（:184-191），与官方逐模板前向一致。
  光学特征只算一次（ResNet_Opt 分支与模板无关）。
- 权重：仓库自带 weights/Pre-training-weight.pth（{epoch, state_dict}，strict 全匹配）。训练集未说明。
- 依赖：timm 只为 DropPath（drop_path=0，推理恒等），没装时打桩；utils/__init__.py:2 导入不存在的 noise_function，
  所以按文件路径单独加载 match_template_function.py。fft_match_batch 只在方形输入下正确（:10 w/h 混用），这里都是方形。
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from .base import load_ckpt


def _stub_timm():
    try:
        from timm.models.layers import DropPath  # noqa: F401
        return False
    except Exception:
        import torch.nn as nn

        class DropPath(nn.Identity):
            def __init__(self, *a, **k):
                super().__init__()

        for name in ("timm", "timm.models", "timm.models.layers"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["timm.models.layers"].DropPath = DropPath
        return True


class RMSOAdapter:
    def __init__(self, repo, weights, device="cuda", template=256, stride=32, chunk=16):
        repo = Path(repo)
        stubbed = _stub_timm()
        sys.path.insert(0, str(repo))
        for k in [k for k in sys.modules if k in ("model", "utils") or k.startswith(("model.", "utils."))]:
            del sys.modules[k]
        from model.RMSO_ConvNeXt import L2NormDense, RMSO_ConvNeXt

        spec = importlib.util.spec_from_file_location("rmso_match", repo / "utils" / "match_template_function.py")
        mt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mt)
        self.fft_match_batch = mt.fft_match_batch
        self.l2 = L2NormDense()
        self.model = RMSO_ConvNeXt()
        self.model.load_state_dict(load_ckpt(weights)["state_dict"], strict=True)
        self.model = self.model.eval().to(device)
        self.device, self.T, self.stride, self.chunk = device, template, stride, chunk
        self.notes = (f"template-grid (ours, not official): T={template}, stride={stride}, full 512 reference, "
                      f"kp = template centre, conf = -min SSD; timm stub={stubbed}")

    def match(self, opt, sar):
        import torch

        H, W = opt.shape
        T, st = self.T, self.stride
        m = self.model
        t_opt = torch.from_numpy(np.ascontiguousarray(opt, np.float32))[None, None].to(self.device)
        t_sar = torch.from_numpy(np.ascontiguousarray(sar, np.float32))
        pos = [(sx, sy) for sy in range(0, sar.shape[0] - T + 1, st) for sx in range(0, sar.shape[1] - T + 1, st)]
        kp0, kp1, conf = [], [], []
        with torch.no_grad():
            f_opt = self.l2(m.ResNet_Opt(m.input_norm(t_opt)))
            for i in range(0, len(pos), self.chunk):
                p = pos[i:i + self.chunk]
                crops = torch.stack([t_sar[sy:sy + T, sx:sx + T] for sx, sy in p])[:, None].to(self.device)
                f_t = self.l2(m.ResNet_Sar(m.input_norm(crops)))
                ssd = self.fft_match_batch(f_opt.expand(len(p), -1, -1, -1), f_t)   # (b, H-T+1, W-T+1)
                flat = ssd.reshape(len(p), -1)
                val, idx = flat.min(dim=1)
                ys, xs = np.unravel_index(idx.cpu().numpy(), ssd.shape[1:])
                c = (T - 1) / 2.0
                for (sx, sy), x, y, v in zip(p, xs, ys, val.cpu().numpy()):
                    kp0.append((x + c, y + c))
                    kp1.append((sx + c, sy + c))
                    conf.append(-float(v))
        return np.asarray(kp0, float), np.asarray(kp1, float), np.asarray(conf, float)
