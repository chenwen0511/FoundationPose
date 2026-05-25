#!/usr/bin/env python3
"""使用 test/ 样例数据调用 FoundationPose HTTP /infer 接口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "test/20260507_105248_1d1db1bb/inputs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Call FoundationPose /infer with test sample")
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="directory containing rgb.png, depth.png, camera.json",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8002/infer",
        help="POST /infer endpoint",
    )
    args = parser.parse_args()

    sample = args.sample_dir.expanduser().resolve()
    rgb = sample / "rgb.png"
    depth = sample / "depth.png"
    camera = sample / "camera.json"
    for p in (rgb, depth, camera):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 1

    try:
        import requests
    except ImportError:
        print("pip install requests", file=sys.stderr)
        return 1

    with rgb.open("rb") as fr, depth.open("rb") as fd, camera.open("rb") as fc:
        files = {
            "rgb": ("rgb.png", fr, "image/png"),
            "depth": ("depth.png", fd, "image/png"),
            "camera": ("camera.json", fc, "application/json"),
        }
        print(f"POST {args.url}")
        print(f"  rgb={rgb}")
        print(f"  depth={depth}")
        print(f"  camera={camera}")
        resp = requests.post(args.url, files=files, timeout=600)

    print(f"status={resp.status_code}")
    try:
        payload = resp.json()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)
        return 1 if not resp.ok else 0

    if not resp.ok:
        return 1

    result_dir = payload.get("result_dir")
    if result_dir:
        print(f"\nresults: {result_dir}/results/")
        for name in (
            "vlm_roi.json",
            "detection_ism.json",
            "detection_ism_raw.json",
            "detection_ism_filtered.json",
            "vis_ism.png",
            "vis_ism_raw.png",
            "vis_sam3_seg.png",
            "vis_pose.png",
            "detection_pose.json",
        ):
            p = Path(result_dir) / "results" / name
            if p.is_file():
                print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
