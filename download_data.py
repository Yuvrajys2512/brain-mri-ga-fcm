"""
Download a subset of the LGG brain-MRI segmentation dataset (Buda et al., the
Kaggle "lgg-mri-segmentation" / kaggle_3m set) from a public Hugging Face mirror.

Each patient folder holds 256x256 slices (*.tif, 3 channels: pre-contrast,
FLAIR, post-contrast) and expert tumour masks (*_mask.tif).

    python download_data.py               # 20 patients (~780 slices, ~200 MB)
    python download_data.py --patients 40
"""

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "Ehsan-rmz/lgg-mri-segmentation-research"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lgg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with urllib.request.urlopen(API) as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    files = [f for f in files if f.startswith("data/") and f.endswith(".tif")]
    patients = sorted({f.split("/")[1] for f in files})
    random.Random(args.seed).shuffle(patients)
    chosen = set(patients[:args.patients])
    todo = [f for f in files if f.split("/")[1] in chosen]
    print(f"{len(patients)} patients available; downloading {len(chosen)} ({len(todo)} files) to {OUT}")

    def fetch(f):
        dst = os.path.join(OUT, *f.split("/")[1:])
        if os.path.exists(dst):
            return None
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(RAW + urllib.parse.quote(f), dst + ".part")
                os.replace(dst + ".part", dst)       # never leave half-written files behind
                return None
            except Exception as e:                   # flaky network: back off and retry
                err = e
                time.sleep(2 * (attempt + 1))
        return f"{f}: {err}"

    failed = []
    with ThreadPoolExecutor(8) as ex:
        for i, res in enumerate(ex.map(fetch, todo), 1):
            if res:
                failed.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    if failed:
        print(f"{len(failed)} files failed (run the script again to retry):")
        for f in failed[:10]:
            print("  ", f)
    else:
        print("done")


if __name__ == "__main__":
    main()
