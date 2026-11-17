"""#270 MATLAB input preparation and strict result collection; run on server.

python runs/B0/code/traditional270.py prep --method rift --split val --io /opt/tmp/baseline270
python runs/B0/code/traditional270.py collect --method rift --split val --io /opt/tmp/baseline270

This does not launch MATLAB or mark an unavailable environment as model failure.
Raw outputs go to B0/raw, then use baselines.fit with the unchanged protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np

from baselines.data import Data
from baselines.match import env_info, git_head, key
from moonlib import inputs
from workbench.launch import require_clean


def linear_sar(sar):
    """Nonnegative linear S1 scaled by its maximum; no dB/clipping search."""
    s1 = sar[..., 0].astype(np.float64) + sar[..., 1].astype(np.float64)
    s1 = np.where(np.isfinite(s1), np.maximum(s1, 0), 0)
    return (s1 / max(float(s1.max()), 1e-12)).astype(np.float32)


def main():
    from scipy.io import loadmat, savemat

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=['prep', 'collect'])
    ap.add_argument('--method', required=True, choices=['rift', 'ossift'])
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--data', default=os.environ.get('MOON_DATA'))
    ap.add_argument('--io', required=True)
    ap.add_argument('--limit', type=int, default=0, help='Debug only; never fit partial formal results')
    ap.add_argument('--selfcheck', action='store_true', help='Optical shifted by (7,-5), validation only')
    args = ap.parse_args()
    require_clean(REPO)
    cfg = json.loads((REPO / 'runs/B0/configs' / (args.method + '_270.json')).read_text(encoding='utf-8'))
    if git_head(REPO / cfg['repo']) != cfg['upstream_commit']:
        raise RuntimeError('Upstream checkout differs from the pinned config, or is dirty')
    ds = Data(args.data)
    pairs = ds.pairs(args.split)[:args.limit or None]
    tag = args.split + ('_selfcheck' if args.selfcheck else '')
    base = Path(args.io) / args.method / tag
    if args.step == 'prep':
        (base / 'in').mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            opt = inputs.optical_div255(ds.optical(args.split, pair))
            if args.selfcheck:
                if args.split != 'val' or not args.limit:
                    raise ValueError('selfcheck requires val and a positive --limit')
                import cv2
                sar = cv2.warpAffine(opt, np.float32([[1, 0, 7], [0, 1, -5]]), opt.shape[::-1])
            else:
                raw = ds.sar(args.split, pair)
                sar = linear_sar(raw) if args.method == 'ossift' else inputs.sar_p2p98(raw)
            savemat(base / 'in' / (key(pair) + '.mat'), {'opt': opt, 'sar': sar})
        (base / 'manifest.json').write_text(json.dumps({'pairs': pairs, 'config': cfg,
            'commit': git_head(REPO), 'selfcheck': args.selfcheck}, indent=1), encoding='utf-8')
        print('prepared', len(pairs), base)
        return
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['pairs'] != pairs or manifest['config'] != cfg:
        raise ValueError('Input manifest differs from requested split/config')
    missing = [p for p in pairs if not (base / 'out' / (key(p) + '.mat')).exists()]
    if missing:
        raise RuntimeError('MATLAB outputs missing: %d; no predictions written' % len(missing))
    store, records = {}, []
    for pair in pairs:
        result = loadmat(base / 'out' / (key(pair) + '.mat'))
        err = str(result['err'][0]) if result['err'].size else ''
        rec = {'pair': pair, 'sec': float(result['sec'].ravel()[0])}
        if err:
            rec['error'] = err
        else:
            p1 = np.asarray(result['p1'], dtype=float).reshape(-1, 2) - 1
            p2 = np.asarray(result['p2'], dtype=float).reshape(-1, 2) - 1
            if len(p1) != len(p2) or not np.isfinite(np.c_[p1, p2]).all():
                raise ValueError('Invalid coordinates for ' + pair)
            store[key(pair)] = np.c_[p1, p2, np.ones(len(p1))].astype(np.float32)
            rec['n'] = len(p1)
            if args.selfcheck:
                rec['median_shift_residual'] = (float(np.median(np.linalg.norm(p2-p1-[7, -5], axis=1)))
                                                if len(p1) else None)
        records.append(rec)
    dest = REPO / 'runs/B0/raw' / (args.method + ('_selfcheck' if args.selfcheck else ''))
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / (args.split + '.npz'), **store)
    (dest / (args.split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    meta = {'config': cfg, 'commit': git_head(REPO), 'repo_commit': git_head(REPO / cfg['repo']),
            'env': env_info(), 'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_manifest': manifest, 'n_pairs': len(pairs), 'n_errors': sum('error' in r for r in records),
            'notes': 'MATLAB 1-based converted to 0-based, before all FSC/CSC geometry; confidence unavailable'}
    (dest / (args.split + '.meta.json')).write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print('collected', dest, len(records))
    if args.selfcheck:
        print(json.dumps(records, indent=1))


if __name__ == '__main__':
    main()
