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
    print(output, flush=True)


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
