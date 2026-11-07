"""单机多卡数据并行（「单卡提速与多卡并行」#226）。[run] ddp = N > 0 时 finetune.train 由 torchrun 起 N 个进程，
每个进程一张卡、每步算 1 个 batch；全部反向做完之后，各卡的梯度取平均（all-reduce），再各自做同一次参数更新。

不用 torch.nn.parallel.DistributedDataParallel 包装模型：RoMa RL 的一步里反向分好几段（细化各级在训练成分内部各自
backward，最后粗级与替身项再 backward 一次，见 finetune/parts/rloo.py），DDP 的梯度钩子要求每次前向只对应一次反向。
这里等全部反向做完再平均，算出来的就是 DDP 的梯度（各卡梯度之和 ÷ 卡数）。

某个参数在一些卡上这一步没有梯度（.grad 为 None，例如那张卡的图对拟合不出参照、细化层没有收到梯度）时：只要有一张卡
有，没有的卡按 0 参与平均，与单卡上一个 batch 里几对图的损失取平均时相同；全部卡都没有时保持 None（AdamW 跳过它）。

单进程时（没有 torchrun 的环境变量）下面的函数都退化为单卡行为。
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def init(n: int) -> int:
    """[run] ddp = n > 0 时初始化进程组（NCCL），把本进程绑到 LOCAL_RANK 那张卡；返回 rank。n = 0 时什么也不做、返回 0。
    torchrun 起的进程数与 n 不同时报错。"""
    if n <= 0:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("用 torchrun 起了多个进程，但配置里 [run] ddp = 0")
        return 0
    ws = int(os.environ.get("WORLD_SIZE", "0"))
    if ws != n:
        raise RuntimeError(f"[run] ddp = {n}，但 WORLD_SIZE = {ws}（应由 torchrun --nproc_per_node={n} 启动）")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    return dist.get_rank()


def close():
    if world() > 1:
        dist.destroy_process_group()


@torch.no_grad()
def broadcast(tensors):
    """rank 0 的值广播给全部进程（训练开始前对齐参数）。"""
    if world() == 1:
        return
    for t in tensors:
        dist.broadcast(t.data, 0)


def _buckets(tensors):
    by = {}
    for t in tensors:
        by.setdefault((t.dtype, t.device), []).append(t)
    return by.values()


@torch.no_grad()
def average_grads(params):
    """各卡梯度取平均（见模块说明）。单进程时什么也不做。"""
    n = world()
    if n == 1:
        return
    dev = params[0].device
    has = torch.tensor([p.grad is not None for p in params], device=dev, dtype=torch.int32)
    dist.all_reduce(has, op=dist.ReduceOp.MAX)
    live = []
    for p, h in zip(params, has.tolist()):
        if h:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            live.append(p)
    for group in _buckets(live):
        flat = torch.cat([p.grad.reshape(-1) for p in group])
        dist.all_reduce(flat)
        flat.div_(n)
        i = 0
        for p in group:
            k = p.numel()
            p.grad.copy_(flat[i:i + k].view_as(p.grad))
            i += k


@torch.no_grad()
def mean_norms(params, groups, mask, scale=1.0, chunk=1 << 24) -> list[float]:
    """几组梯度各自「各卡取平均之后」的范数（量真正施加到参数上的梯度大小用；各卡范数的平均不等于平均梯度的范数，
    它还取决于各卡梯度方向是否一致）。params：参数列表（各卡同序，只用来取形状）；groups：K 组与 params 等长的梯度列表，
    元素为 None 时按 0；mask：本卡要算进范数的参数（bool 列表），任一卡为真的参数各卡都算进去。各组先除以 scale
    （本卡的 GradScaler 因子）再平均。返回 K 个范数。全部进程都要调用（内含 all-reduce）；单进程时即本卡的范数。"""
    n = world()
    dev = params[0].device if params else torch.device("cpu")
    live = list(mask)
    if n > 1:
        has = torch.tensor([bool(m) for m in mask], device=dev, dtype=torch.int32)
        dist.all_reduce(has, op=dist.ReduceOp.MAX)
        live = [bool(h) for h in has.tolist()]
    sq = torch.zeros(len(groups), dtype=torch.float64, device=dev)
    idx = [i for i, h in enumerate(live) if h]

    def flush(batch):
        flat = torch.stack([torch.cat([(torch.zeros(params[i].numel(), device=dev) if g[i] is None
                                        else g[i].detach().reshape(-1).float().to(dev)) for i in batch])
                            for g in groups]) / scale
        if n > 1:
            dist.all_reduce(flat)
            flat.div_(n)
        sq.add_(flat.double().square().sum(1))

    batch, size = [], 0
    for i in idx:
        batch.append(i)
        size += params[i].numel()
        if size >= chunk:
            flush(batch)
            batch, size = [], 0
    if batch:
        flush(batch)
    return [float(x) for x in sq.sqrt()]


def gather(obj) -> list:
    """每个进程的 obj（可 pickle）→ 按 rank 排的列表；单进程时 [obj]。"""
    if world() == 1:
        return [obj]
    out = [None] * world()
    dist.all_gather_object(out, obj)
    return out


@torch.no_grad()
def max_diff(tensors) -> float:
    """各进程上同名张量与 rank 0 的最大绝对差（检查各卡参数是否一致）；单进程时 0。"""
    if world() == 1:
        return 0.0
    worst = torch.zeros((), device=tensors[0].device, dtype=torch.float64)
    for group in _buckets(tensors):
        flat = torch.cat([t.detach().reshape(-1) for t in group])
        ref = flat.clone()
        dist.broadcast(ref, 0)
        worst = torch.maximum(worst, (flat.double() - ref.double()).abs().max())
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    return float(worst)


def mean_stats(rows: list[dict]) -> dict:
    """各卡的日志量 → 逐键取平均（只对有这个键的卡平均；值都是数）。"""
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = {}
    for k in keys:
        v = [float(r[k]) for r in rows if k in r]
        out[k] = round(sum(v) / len(v), 5)
    return out
