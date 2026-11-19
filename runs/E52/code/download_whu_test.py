"""Resume downloading the author's WHU-OPT-SAR Test pairs and record verified TIFFs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen
import zlib

SOURCE_MANIFEST = Path(__file__).with_name("whu_test_sources.json")
MIRROR_MANIFEST = Path(__file__).with_name("whu_test_mirror_ranges.json")


def inspect(path, modality):
    import rasterio

    with rasterio.open(path) as source:
        expected_bands = {"optical": 4, "sar": 1}[modality]
        if (source.width, source.height, source.count) != (5556, 3704, expected_bands):
            raise ValueError(f"Unexpected TIFF dimensions or band count: {path}")
        for _, window in source.block_windows(1):
            source.read(window=window)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def mirror_crc(path, entry):
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            crc = zlib.crc32(block, crc)
    if path.stat().st_size != entry["tiff_bytes"] or f"{crc:08x}" != entry["crc32"]:
        raise ValueError(f"WHU mirror checksum or byte length differs for {path}")


def fetch_mirror(path, link, entry):
    partial = path.with_suffix(".hf.part")
    start, end = entry["data_offset"], entry["data_end_inclusive"]
    try:
        request = Request(link, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(request, timeout=90) as response, partial.open("wb") as stream:
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                raise ValueError(f"Wrong mirror byte range for {path.name}: {content_range}")
            decoder = zlib.decompressobj(-zlib.MAX_WBITS)
            for block in iter(lambda: response.read(1 << 20), b""):
                stream.write(decoder.decompress(block))
            stream.write(decoder.flush())
            if not decoder.eof or decoder.unused_data:
                raise ValueError(f"Incomplete compressed member for {path.name}")
    except OSError as error:
        # urllib errors can include the temporary access URL.
        raise RuntimeError(f"Mirror transfer failed for {path.name}: {type(error).__name__} "
                           f"{getattr(error, 'code', '')}") from None
    mirror_crc(partial, entry)
    return partial


def fetch(root, filename, modality, file_id, proxy, verify_only, mirror):
    path = root / modality / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    source = "existing"
    if not path.exists():
        if verify_only:
            raise FileNotFoundError(path)
        if mirror:
            link, entry = mirror[modality][0], mirror[modality][1]["records"][filename]
            partial = fetch_mirror(path, link, entry)
            source = mirror[modality][1]["url"]
        else:
            partial = path.with_suffix(".part")
            url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            command = ["curl", "--fail", "--location", "--silent", "--show-error",
                       "--connect-timeout", "20", "--max-time", "1800",
                       "--continue-at", "-", "--output", str(partial), url]
            if proxy:
                command[1:1] = ["--proxy", proxy]
            subprocess.run(command, check=True)
            source = "Google Drive"
        info = inspect(partial, modality)
        partial.replace(path)
    else:
        if mirror:
            mirror_crc(path, mirror[modality][1]["records"][filename])
        info = inspect(path, modality)
    print(f"{modality}/{filename} {info['bytes']} bytes", flush=True)
    return {"drive_id": file_id, "source": source, **info}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--proxy")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--mirror-links", type=Path, help="JSON with temporary direct CDN links")
    args = parser.parse_args()
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    mirror = None
    if args.mirror_links:
        links = json.loads(args.mirror_links.read_text(encoding="utf-8"))
        members = json.loads(MIRROR_MANIFEST.read_text(encoding="utf-8"))
        mirror = {m: (links[m], members[m]) for m in links}
    verified, errors = {}, {}
    with ThreadPoolExecutor(max_workers=3) as workers:
        jobs = {
            (filename, modality): workers.submit(fetch, args.root, filename, modality, file_id,
                                                 args.proxy, args.verify_only, mirror)
            for filename, pair in manifest["test"].items() for modality, file_id in pair.items()
        }
        for (filename, modality), job in jobs.items():
            try:
                verified.setdefault(filename, {})[modality] = job.result()
            except Exception as error:
                errors[f"{modality}/{filename}"] = str(error)
    report = {**manifest, "verified": verified, "errors": errors}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)
    print(f"{sum(len(pair) for pair in verified.values())}/{len(jobs)} TIFFs verified", flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
