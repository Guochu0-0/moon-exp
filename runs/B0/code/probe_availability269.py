"""Record tiny anonymous publication-endpoint probes, never download model weights.

ProxyHandler({}) avoids silently borrowing an undocumented proxy. A reachable
HTML share page is not evidence that a checkpoint has actually been obtained.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.error
import urllib.request


def main():
    endpoints = {
        "fhreg_google_folder": "https://drive.google.com/embeddedfolderview?id=1T1Ic0uxSIBx_qdQJQEV68jzGlwBFR3TU",
        "vmgga_baidu_share": "https://pan.baidu.com/s/1udp4OAzw_EL9QEI3FyG4Wg?pwd=bf8d",
        "mifnet_dropbox_folder": "https://www.dropbox.com/scl/fo/y9sriuno6xi9axuhb184c/AMR6jdKuOh_fZoI-r8rU2u4?rlkey=bh6ufuj2o9kms252cugiy1ub9&st=6bxu4pd8&dl=0",
        "mifnet_sd21_official_api": "https://huggingface.co/api/models/stabilityai/stable-diffusion-2-1",
        "mifnet_sd21_mirror_api": "https://hf-mirror.com/api/models/stabilityai/stable-diffusion-2-1",
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for name, url in endpoints.items():
        record = {"name": name, "url": url, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "anonymous": True, "proxy": "disabled", "weight_download_attempted": False}
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(request, timeout=8) as response:
                data = response.read(32768)
                text = data.decode("utf-8", errors="replace")
                title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
                record.update(http_status=response.status, content_type=response.headers.get("Content-Type"),
                              bytes_read=len(data), page_title=title.group(1).strip() if title else None,
                              checkpoint_name_mentioned="fhreg_guso.ckpt" in text)
        except urllib.error.HTTPError as error:
            record.update(http_status=error.code, error=str(error),
                          error_body_excerpt=error.read(256).decode("utf-8", errors="replace"))
        except Exception as error:
            record.update(http_status=None, error=type(error).__name__ + ": " + str(error))
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    dest = Path("runs/B0/extra/availability269.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"probes": records}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
