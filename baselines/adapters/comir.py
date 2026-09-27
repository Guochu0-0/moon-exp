"""CoMIR（MIDA-group/CoMIR，NeurIPS 2020，third_party/CoMIR，1750b11）。

它不是匹配器，而是**表示学习 + 经典配准**：两个模态各一个 DenseUNet（tiramisu），对比学习把两者映射到共享表示
（CoMIR），再在 CoMIR 上跑单模态的 SIFT。这里是 zero-shot：用官方 Release 1.0（results_models.zip）里的 Zurich 模型
（航拍 RGB↔NIR），不在月球数据上训练。

对照官方：
- 模型：`ModNet(DenseUNet)`（train-zurich.ipynb cell 10），include_top=False + 无 bias 的 1×1 conv 到 latent_channels=3；
  结构参数 `tiramisu_args`（cell 3）。Release 里的 .pt 是 `torch.save({"modelA": module, "modelB": module})` 整个模块
  pickle（cell 13），类引用 `__main__.ModNet`。这里用自定义 unpickler 把它映射到本文件的 ModNet，只取 state_dict，
  再按 cell 3 的参数重建、strict=True 加载（老 pickle 的 UpsamplingNearest2d 缺新版 torch 的属性，不直接用）。
- 权重：`model_zurich_rot90_enabled.pt`——cell 21「Checking the registration」加载的那个，开了 C4 等变（论文主推）。
- 输入：A = RGB 3 通道、B = NIR 1 通道，官方是 12 位原始值 /2¹²（cell 5）。这里光学灰度复制成 3 通道送 A，SAR 送 B，
  值用主表统一映射的 float [0,1]（不做 /2¹² 的对应换算，属于域差的一部分）。512 是 128 的倍数，官方导出按 128 裁
  （train-biodata.ipynb 末 cell），这里无需裁。eval 模式。
- 配准：官方用 Fiji v1.52p 跑 scripts/compute_sift.py（readme「Part 2」）：mpicbg `FloatArray2DSIFT`，
  fdSize 4、fdBins 8、maxOctaveSize 1024、minOctaveSize 128、steps 4、initialSigma 1.6（:65-71）；
  `createMatches(f1, f2, rod=0.9)`（:94-97），之后 RigidModel2D RANSAC（:100-109）。这里经 JPype 调同一个 mpicbg
  实现，取 RANSAC **之前**的 createMatches 结果，交给统一的仿射 RANSAC。
- 通道：官方 CoMIR 存成 H×W×3 float tif（train-biodata.ipynb 末 cell），Fiji `IJ.openImage(...).getProcessor()`
  （compute_sift.py:58,82）只取当前 slice，即第 0 通道。**推断**，未在 Fiji 里核实；`channel` 参数可改。
  像素值原样送 SIFT（`convertToFloat` 对 float 图不做变换），不归一化——mpicbg 的对比度阈值与强度尺度相关，照官方。
- 坐标：mpicbg `Feature.location` = 所在 octave 坐标 × 2^o（FloatArray2DSIFT.java:472），octave 0 即原图、不放大
  （512 < maxOctaveSize）。整数 = 像素中心，与本仓库约定一致。createMatches 不给分数，conf 为 None。
- 依赖：JPype1 + JDK，classpath 放 mpicbg（ij、jama 是它的依赖）。版本见配置。
"""
from __future__ import annotations

import glob
import pickle
import sys
import types

import numpy as np

# train-zurich.ipynb cell 3
TIRAMISU_ARGS = {
    "init_conv_filters": 16, "down_blocks": (2, 2, 2), "up_blocks": (2, 2, 2), "bottleneck_layers": 3,
    "upsampling_type": "upsample", "transition_pooling": "max", "dropout_rate": 0.0, "early_transition": False,
    "activation_func": None, "compression": 1.0, "efficient": False,
}
LATENT_CHANNELS = 3
# compute_sift.py:65-71, :94
SIFT_PARAMS = {"fdSize": 4, "fdBins": 8, "maxOctaveSize": 1024, "minOctaveSize": 128, "steps": 4, "initialSigma": 1.6}
ROD = 0.9


def _modnet_class(DenseUNet):
    class ModNet(DenseUNet):  # train-zurich.ipynb cell 10
        def __init__(self, **args):
            super().__init__(**args, include_top=False)
            out_channels = self.get_channels_count()[-1]
            self.final_conv = __import__("torch").nn.Conv2d(out_channels, LATENT_CHANNELS, 1, bias=False)

        def forward(self, x):
            return self.final_conv(super().forward(x))

    return ModNet


def _pickle_module(ModNet):
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "ModNet":
                return ModNet
            return super().find_class(module, name)

    mod = types.ModuleType("comir_pickle")
    mod.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
    mod.Unpickler = Unpickler
    return mod


def _start_jvm(classpath):
    import jpype

    if not jpype.isJVMStarted():
        jars = sorted(glob.glob(classpath)) if any(c in classpath for c in "*?[") else [classpath]
        if not jars:
            raise FileNotFoundError(f"mpicbg classpath 为空：{classpath}")
        jpype.startJVM(classpath=jars, convertStrings=False)
    return jpype


class CoMIRAdapter:
    def __init__(self, repo, weights, device="cuda", channel=0, classpath="/opt/mpicbg/*"):
        import torch

        sys.path.insert(0, str(repo))
        for k in [k for k in sys.modules if k in ("models", "utils") or k.startswith(("models.", "utils."))]:
            del sys.modules[k]
        from models.tiramisu import DenseUNet

        ModNet = _modnet_class(DenseUNet)
        kw = {"map_location": "cpu", "pickle_module": _pickle_module(ModNet)}
        if "weights_only" in torch.load.__code__.co_varnames:
            kw["weights_only"] = False
        ckpt = torch.load(str(weights), **kw)
        self.models = []
        for key, cin in (("modelA", 3), ("modelB", 1)):
            m = ModNet(in_channels=cin, nb_classes=LATENT_CHANNELS, **TIRAMISU_ARGS)
            m.load_state_dict(ckpt[key].state_dict(), strict=True)
            self.models.append(m.eval().to(device))
        self.device, self.channel = device, channel

        jpype = _start_jvm(classpath)
        self._jp = jpype
        self.SIFT = jpype.JClass("mpicbg.imagefeatures.FloatArray2DSIFT")
        self.FloatArray2D = jpype.JClass("mpicbg.imagefeatures.FloatArray2D")
        self.param = self.SIFT.Param()
        for k, v in SIFT_PARAMS.items():
            setattr(self.param, k, v)
        mp_ver = jpype.JClass("mpicbg.imagefeatures.FloatArray2DSIFT").class_.getPackage().getImplementationVersion()
        self.notes = (f"zero-shot Zurich weights (RGB<-optical gray x3, NIR<-SAR, [0,1] not /2^12); "
                      f"SIFT on CoMIR channel {channel} via mpicbg {mp_ver} (JPype), params {SIFT_PARAMS}, rod {ROD}; "
                      f"raw createMatches before official RigidModel2D RANSAC")

    def comir(self, opt, sar):
        import torch

        a = torch.from_numpy(np.repeat(np.ascontiguousarray(opt, np.float32)[None], 3, 0))[None].to(self.device)
        b = torch.from_numpy(np.ascontiguousarray(sar, np.float32))[None, None].to(self.device)
        with torch.no_grad():
            la, lb = self.models[0](a), self.models[1](b)
        return la[0].cpu().numpy(), lb[0].cpu().numpy()     # (3, H, W) each

    def _features(self, img):
        h, w = img.shape
        arr = self._jp.JArray(self._jp.JFloat)(np.ascontiguousarray(img, np.float32).ravel())
        sift = self.SIFT(self.param)
        sift.init(self.FloatArray2D(arr, w, h))
        return sift.run()

    def match(self, opt, sar):
        la, lb = self.comir(opt, sar)
        f0, f1 = self._features(la[self.channel]), self._features(lb[self.channel])
        pms = self.SIFT.createMatches(f0, f1, self._jp.JFloat(ROD))
        kp0 = np.array([list(pm.getP1().getL()) for pm in pms], float).reshape(-1, 2)
        kp1 = np.array([list(pm.getP2().getL()) for pm in pms], float).reshape(-1, 2)
        return kp0, kp1, None
