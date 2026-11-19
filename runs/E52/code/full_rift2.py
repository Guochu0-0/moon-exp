"""Run pinned RIFT2 detection in an authorized MATLAB process."""

import argparse
from pathlib import Path
import sys

from workbench import launch
from .repeatability import RUN

ROOT = RUN.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=("moon", "whu"))
    parser.add_argument("detections", type=Path)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()

    import matlab.engine

    directory = args.detections.resolve() / args.domain
    manifest = directory / "manifest.json"
    if not manifest.is_file() or not any((directory / "rift2_input").glob("*.mat")):
        raise FileNotFoundError(f"Missing prepared RIFT2 inputs in {directory}")
    name = f"full_{args.domain}_rift2_matlab"
    launch.begin(RUN, name, [sys.executable, "-m", __spec__.name, *sys.argv[1:]], threads=2)
    try:
        session = matlab.engine.connect_matlab(args.session)
        version = session.version()
        code = str(Path(__file__).parent)
        upstream = str(ROOT / "third_party" / "RIFT2")
        # Scope MATLAB's path to this call, including when detection fails.

        def matlab_string(value):
            return "'" + str(value).replace("'", "''") + "'"

        paths = ", ".join(map(matlab_string, (code, upstream)))
        directories = ", ".join(map(matlab_string, (directory / "rift2_input", directory / "rift2")))
        code = ("e52_previous_path = path; try; "
                f"addpath({paths}); rift2_detect_batch({directories}); "
                "catch e52_error; path(e52_previous_path); rethrow(e52_error); end; path(e52_previous_path);")
        session.eval(code, nargout=0)
    except BaseException:
        launch.end(RUN, name, "failed")
        raise
    else:
        launch.end(RUN, name, "ok", matlab=version)


if __name__ == "__main__":
    main()
