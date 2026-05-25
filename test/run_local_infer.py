#!/usr/bin/env python3
"""离线运行 FoundationPose pipeline（不启动 HTTP），使用 test/ 白盘样例。"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = ROOT / "test/20260507_105248_1d1db1bb/inputs"
DEFAULT_MESH = ROOT / "test/CAD/tray_180mm_centered_mesh_v2.ply"
DEFAULT_MESH_SCALE = 0.001
DEFAULT_SAM3_PROMPT = "Plastic Reel"
DEFAULT_SAM3_ROOT = ROOT.parent / "sam3" if (ROOT.parent / "sam3").is_dir() else Path("/home/ubuntu/stephen/01-code/sam3")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local FoundationPose infer on white-plate test sample")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--mesh-file", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--mesh-scale", type=float, default=DEFAULT_MESH_SCALE)
    parser.add_argument("--sam3-prompt", default=DEFAULT_SAM3_PROMPT)
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="disable VLM ROI filter (GENPOSE2_USE_VLM_ROI_FILTER=0)",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "service_outputs")
    args = parser.parse_args()

    sample = args.sample_dir.expanduser().resolve()
    rgb_path = sample / "rgb.png"
    depth_path = sample / "depth.png"
    camera_path = sample / "camera.json"
    mesh_path = args.mesh_file.expanduser().resolve()

    for p in (rgb_path, depth_path, camera_path, mesh_path):
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            return 1

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import os

    os.environ.setdefault("GENPOSE2_SAM3_ROOT", str(DEFAULT_SAM3_ROOT.resolve()))
    os.environ["FOUNDATIONPOSE_MESH_FILE"] = str(mesh_path)
    os.environ["FOUNDATIONPOSE_MESH_SCALE"] = str(args.mesh_scale)
    os.environ["GENPOSE2_SAM3_PROMPT"] = args.sam3_prompt
    if args.no_vlm:
        os.environ["GENPOSE2_USE_VLM_ROI_FILTER"] = "0"
        os.environ["GENPOSE2_USE_VLM_ROI"] = "0"
    else:
        os.environ.setdefault("GENPOSE2_USE_VLM_ROI_FILTER", "1")
        os.environ.setdefault("GENPOSE2_USE_VLM_ROI", "1")

    from seg.sam3_seg import _sam3_infer_script, _sam3_python, _validate_sam3_toolchain
    from seg.vlm_seg import use_vlm_roi_filter

    py = _sam3_python()
    script = _sam3_infer_script()
    try:
        _validate_sam3_toolchain(py, script)
        print(f"SAM3 toolchain: ok python={py} infer={script}")
    except FileNotFoundError as exc:
        print(f"SAM3 toolchain missing: {exc}", file=sys.stderr)
        return 1

    from http_server import _load_foundationpose_models, _run_foundationpose_pipeline

    request_id = f"local_test_{uuid.uuid4().hex[:8]}"
    output_dir = args.output_root.expanduser().resolve() / request_id
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    for src, name in ((rgb_path, "rgb.png"), (depth_path, "depth.png"), (camera_path, "camera.json")):
        dst = input_dir / name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())

    print(
        f"mesh={mesh_path} scale={args.mesh_scale} "
        f"sam3_prompt={args.sam3_prompt!r} use_vlm_roi_filter={use_vlm_roi_filter()}"
    )
    _load_foundationpose_models(mesh_path)

    print(f"running pipeline -> {output_dir}")
    payload = _run_foundationpose_pipeline(
        input_dir / "rgb.png",
        input_dir / "depth.png",
        input_dir / "camera.json",
        output_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    results_dir = output_dir / "results"
    print(f"\nvisualizations under: {results_dir}")
    for name in (
        "vlm_roi.json",
        "vlm_roi_vis.png",
        "detection_ism.json",
        "detection_ism_raw.json",
        "detection_ism_filtered.json",
        "vis_ism.png",
        "vis_ism_raw.png",
        "vis_sam3_seg.png",
        "vis_pose.png",
        "detection_pose.json",
    ):
        p = results_dir / name
        if p.is_file():
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
