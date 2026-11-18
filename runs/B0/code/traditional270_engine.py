"""Run one prepared batch in 154's authorized shared MATLAB session."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from workbench import launch


def patch_himo(config, overlay):
    """Apply the two reviewable changes without modifying the upstream checkout."""
    upstream = REPO / config['repo'] / 'HIMO_image_matching' / 'func_HIMO'
    for name in ('Match_Keypoint.m', 'Multiscale_Strategy.m'):
        patch = Path(__file__).with_name(name + '.patch')
        source = Path(overlay) / (name + '.original')
        source.write_text((upstream / name).read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        subprocess.run(['patch', '--batch', '--fuzz=0', '-o', str(Path(overlay) / name),
                        str(source), str(patch)], check=True, timeout=30)


def main():
    import matlab.engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('batch', nargs='?', type=Path)
    args = parser.parse_args()
    session = matlab.engine.connect_matlab(args.session)
    try:
        version = session.version()
        print(version, flush=True)
        if args.batch is None:
            session.eval("matchFeatures(eye(3),eye(3)); imadjust(eye(3)); parallel.Settings;", nargout=0)
            return

        batch = args.batch.resolve()
        manifest = json.loads((batch / 'manifest.json').read_text(encoding='utf-8'))
        state = launch.require_clean(REPO)
        if manifest['commit'] != state['commit']:
            raise RuntimeError('Prepared inputs refer to a different repository commit')
        config = manifest['config']
        method = config['method']
        attempt = f'{method}_selfcheck270' if manifest['selfcheck'] else method
        code = str(Path(__file__).parent)
        session.addpath(code, nargout=0)
        # The recorded command resolves to the pinned code and absolute input/output directories.
        launch.begin(REPO / 'runs/B0', attempt, sys.argv, split=batch.name, config=config,
                     env={'matlab': version, 'threads': 2, 'device': 'cpu', 'precision': 'double'})
        try:
            with tempfile.TemporaryDirectory(prefix='himo270-') as overlay:
                if method == 'himo':
                    patch_himo(config, overlay)
                session.traditional270_split(method, str(REPO), str(batch / 'in'), str(batch / 'out'),
                                             2, overlay, nargout=0)
        except BaseException:
            launch.end(REPO / 'runs/B0', attempt, 'fail')
            raise
        else:
            launch.end(REPO / 'runs/B0', attempt, 'ok')
        finally:
            session.rmpath(code, nargout=0)
    finally:
        session.quit()


if __name__ == '__main__':
    main()
