"""#265 only: evaluate retained final checkpoints in their original nodes.

Run two workers with --worker 0/1, then --report after both finish.
Existing best-checkpoint predictions are preserved. No training is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
Y = Path('/remote-home/xufang/YGC')
SOURCE = Y / 'moon-exp' / 'runs'
WB = '/opt/envs/wb/bin/python'
TASKS = [
    ('B0m', 'anymatch_roma__minmax', None, None, 'anymatch_roma__minmax'),
    ('E41', 'main__screen0.8', 'main__screen0.8', 7907, 'anymatch_roma__minmax'),
    ('T13', 'main__step8000', 'main', 8000, 'anymatch_roma__minmax'),
    ('S1', 'main__step8000', 'main', 8000, 'anymatch_loftr'),
    ('S1', 'seed1__step8000', 'seed1', 8000, 'anymatch_loftr'),
    ('S1', 'seed2__step8000', 'seed2', 8000, 'anymatch_loftr'),
]


def worker(index):
    os.chdir(REPO)
    gpu = os.environ['GPU']
    for i, (rid, method, original, step, config) in enumerate(TASKS):
        if i % 2 != index:
            continue
        run = REPO / 'runs' / rid
        lock = run / 'ckpt' / f'test265_{method}.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.mkdir()  # Refuse duplicate launches, including failed previous attempts.
        if original is not None:
            ckpt = SOURCE / rid / 'ckpt' / original / f'ckpt_{step}.pt'
            if not ckpt.is_file() or ckpt.stat().st_size == 0:
                raise RuntimeError(f'Missing checkpoint: {ckpt}')
        else:
            ckpt = '-'
        if original is not None and method != original:
            src = SOURCE / rid / 'sweep' / original / 'S' / 'preds' / f'step{step}'
            dst = run / 'preds' / method
            dst.mkdir(parents=True, exist_ok=True)
            if not (src / 'val.jsonl').is_file():
                raise RuntimeError(f'Missing retained Val predictions: {src}')
            for name in ('val.jsonl', 'val.meta.json', 'val_matches.npz'):
                if (src / name).is_file():
                    shutil.copy2(src / name, dst / name)
        cmd = ['bash', 'scripts/finetune/eval_ckpt.sh', str(run), method,
               str(ckpt), f'configs/baselines/{config}.json']
        print(f'START gpu={gpu} {rid}/{method}', flush=True)
        subprocess.run(cmd, check=True)
        (lock / 'done').write_text('ok\n')
        print(f'DONE {rid}/{method}', flush=True)


def report():
    import numpy as np
    from workbench.dataset import Dataset
    from workbench.evaluate import pair_errors
    from workbench.records import load_runs

    os.chdir(REPO)
    subprocess.run([WB, '-m', 'workbench', 'eval', 'B0m', 'E41', 'T13', 'S1'], check=True)
    runs = load_runs(REPO / 'runs')
    ds = Dataset(os.environ['MOON_DATA'])
    assert len(ds.labelled('test')) == 953
    # Rows: selection rule, node, method, checkpoint step, retained Val source.
    rows = [('fixed', 'B0m', 'anymatch_roma__minmax', None, None)]
    rows += [('last', 'E41', 'main__screen0.8', 7907, None),
             ('best', 'E41', None, 4250, 'main__screen0.8')]
    for rid, methods in [('T13', ['main', 'main__s1']), ('S1', ['main', 'seed1', 'seed2'])]:
        for m in methods:
            peak = json.loads((SOURCE / rid / 'sweep' / m / 'peak.json').read_text())
            best = int(peak['step'])
            rows.append(('best', rid, m, best, None))
            rows.append(('last', rid, m if best == 8000 else f'{m}__step8000', 8000, None))
    table = []
    for rule, rid, method, step, val_source in rows:
        metrics = json.loads((REPO / 'runs' / rid / 'metrics.json').read_text())['methods']
        if method is None:
            sweep = json.loads((SOURCE / rid / 'sweep' / val_source / 'S' / 'metrics.json').read_text())
            val = sweep['methods'][f'step{step}']['val']['summary']
            test = None
        else:
            val = metrics[method]['val']['summary']
            test = metrics[method].get('test', {}).get('summary')
            if test is None or test['n'] != 953:
                raise RuntimeError(f'Incomplete Test: {rid}/{method}')
        table.append(dict(rule=rule, node=rid, method=method, step=step, val=val, test=test))
    errors = {}
    for rid, m in [('B0m', 'anymatch_roma__minmax'), ('E41', 'main__screen0.8'),
                   ('T13', 'main'), ('T13', 'main__step8000'),
                   ('S1', 'main'), ('S1', 'main__step8000')]:
        preds = runs[rid].preds(m, 'test')
        assert set(preds) == set(ds.labelled('test')), (rid, m)
        errors[(rid, m)] = pair_errors(ds, 'test', preds)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, 953, size=(2000, 953))
    paired = []
    ours = np.clip(1 - errors[('E41', 'main__screen0.8')] / 5, 0, None)
    for rule in ('last', 'best'):
        for rid, base in [('B0m', 'anymatch_roma__minmax'), ('T13', 'main'), ('S1', 'main')]:
            if rule == 'best':
                paired.append(dict(rule=rule, reference=rid, delta=None, ci=None,
                                   missing='E41 ckpt_4250.pt unavailable; no Test inference'))
                continue
            method = f'{base}__step8000' if rid in ('T13', 'S1') else base
            delta = ours - np.clip(1 - errors[(rid, method)] / 5, 0, None)
            ci = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
            paired.append(dict(rule=rule, reference=rid, delta=float(delta.mean()), ci=ci.tolist()))
    output = REPO / 'runs' / 'E41' / 'extra' / 'data' / 'test265.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    from workbench.evaluate import jsonable
    output.write_text(json.dumps(jsonable(dict(rows=table, paired=paired, resamples=2000, seed=0)),
                                 ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    append_notes(table, paired)
    print(output, flush=True)


def append_notes(rows, paired):
    heading = '## 2026-11-17：Test 与选存档规则补评（#265）'
    labels = {'fixed': '固定起点', 'last': '最后存档', 'best': 'Val 最高存档'}
    def metrics(summary):
        return ' / '.join(f'{summary[f"auc@{t}"]:.4f}' for t in (3, 5, 10)) if summary else '缺失'
    def ci(summary):
        return '[{:.4f}, {:.4f}]'.format(*summary['auc@5_ci']) if summary else '缺失'
    for rid in ('B0m', 'E41', 'T13', 'S1'):
        path = REPO / 'runs' / rid / 'notes.md'
        original = path.read_text(encoding='utf-8')
        if heading in original:
            raise RuntimeError(f'Notes already appended: {path}; do not overwrite user edits')
        text = [heading, '',
                '本节只补评已有存档，没有训练。Test 为不加扰动的 953 对有标注 patch 对；'
                '配准误差、AUC、SR、失败率与单方法 95% 自助法区间沿用 '
                '`workbench.protocol`（1000 次重抽，种子 0）。本次新增 Test 推理均在 '
                '160 的 V100 上完成；复用的 Val 与已有最佳 Test 沿用原记录。'
                '推理配置保持原方法不变：长边 640，RoMa 采样 10000 点，RANSAC 3 px；'
                'SCENES 保持 AnyMatch-LoFTR 的原始 p2p98 SAR 映射。', '']
        if rid == 'E41':
            text += ['为补齐论文主表，本表合并起点、自训练、SCENES 与门槛 0.8 种子 0 的两种选存档规则。'
                     'E41 第 4250 次更新的存档已经缺失，只有 Val 预测保留，因此该规则下的 Test '
                     '及配对差无法补评；按票约定没有重训。#264 尚未产生新种子存档，本节没有种子 1、2。', '']
        selected = rows if rid == 'E41' else [r for r in rows if r['node'] == rid]
        text += ['| 节点 / 方法 | 选存档规则 | 更新次数 | Val AUC@3 / 5 / 10 | Test AUC@3 / 5 / 10 | Test AUC@5 95% 区间 |',
                 '|---|---|---:|---|---|---|']
        for r in selected:
            method = r['method'] or 'main__screen0.8（存档缺失）'
            step = str(r['step']) if r['step'] is not None else '—'
            text.append(f"| {r['node']}/{method} | {labels[r['rule']]} | {step} | {metrics(r['val'])} | {metrics(r['test'])} | {ci(r['test'])} |")
        number = {'B0m': 2, 'E41': 7, 'T13': 2, 'S1': 3}[rid]
        text += ['', f'表 {number}：同一节点的两种选存档规则。Test 缺失表示对应模型存档不在，不是配准失败。', '']
        if rid == 'E41':
            text += ['配对自助法按同一批 953 对同步有放回重抽 2000 次，种子 0，差为本文减参照。'
                     '自训练与 SCENES 的配对比较使用种子 0；其余种子只列单方法读数。', '',
                     '| 选存档规则 | 参照 | Test AUC@5 之差 | 95% 区间 |', '|---|---|---:|---|']
            for p in paired:
                delta = f"{p['delta']:+.4f}" if p['delta'] is not None else '缺失'
                interval = '[{:+.4f}, {:+.4f}]'.format(*p['ci']) if p['ci'] else '缺失'
                text.append(f"| {labels[p['rule']]} | {p['reference']} | {delta} | {interval} |")
            text += ['', '表 8：两种规则下本文减参照的 Test AUC@5 配对差。正值表示本文的 AUC@5 较高，'
                     '负值表示较低；区间反映更换同样数量测试对带来的波动，不反映重新训练的随机性。', '',
                     '完整数值（含 AUC@20、SR、失败率）保存在各原节点的 `metrics.json`；'
                     '总表和配对差的数据见 [附件](extra/data/test265.json)。'
                     '评测脚本为 `code/test265.py`，模型与每次启动的提交、主机、卡号见各节点 `launch/`。', '']
        elif rid == 'B0m':
            baseline = json.loads((SOURCE / 'B0' / 'metrics.json').read_text())['methods']['anymatch_roma']['test']['summary']
            current = selected[0]['test']['auc@5']
            old = baseline['auc@5']
            text += [f'起点 min–max 映射 Test AUC@5 为 {current:.4f}，B0 分位数映射为 {old:.4f}'
                     f'（约 {old * 100:.1f}%），差为 {current - old:+.4f}。这里只改变 SAR 输入映射。', '']
        else:
            text += ['跨方法总表与配对差见 [E41 本次补评](../E41/notes.md)。'
                     '若最佳与最后是同一次更新，直接复用同一份预测，没有重复推理。', '']
        path.write_text(original.rstrip() + '\n\n' + '\n'.join(text) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=(0, 1))
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report()
    elif args.worker is not None:
        worker(args.worker)
    else:
        parser.error('choose --worker or --report')
