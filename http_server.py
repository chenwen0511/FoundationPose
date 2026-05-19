"""
FoundationPose HTTP 服务：/infer 接收 rgb、depth、camera 三个 multipart 文件。

流程：SAM3 文本分割（``sam3_seg.py``）→ 对每个实例调用 FoundationPose ``register()`` → 可视化。

启动示例（白盘测试样例）::

    export FOUNDATIONPOSE_MESH_FILE=test/CAD/tray_180mm_centered_mesh_v2.ply
    export FOUNDATIONPOSE_MESH_SCALE=0.001
    export GENPOSE2_SAM3_PROMPT="Plastic Reel"
    export GENPOSE2_SAM3_ROOT=/path/to/sam3
    python http_server.py --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import cv2
import numpy as np
import trimesh

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from sam3_seg import (
    DEFAULT_SAM3_INFER_SCRIPT,
    DEFAULT_SAM3_PROMPT,
    DEFAULT_SAM3_PYTHON,
    Sam3SegmentationResult,
    _sam3_infer_script,
    _sam3_python,
    _sam3_root,
    _validate_sam3_toolchain,
    get_instance_bool_masks,
    run_sam3_segmentation,
    visualize_sam3_ism,
    visualize_sam3_mask_exr,
)

DEFAULT_OUTPUT_ROOT = ROOT_DIR / "service_outputs"
DEFAULT_MESH_FILE = ROOT_DIR / "test/CAD/tray_180mm_centered_mesh_v2.ply"
DEFAULT_SAM3_PROMPT = "Plastic Reel"
DEFAULT_MESH_SCALE = 0.001

_POSE_COLORS_RGB: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (255, 128, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 64, 64),
    (64, 128, 255),
    (128, 255, 128),
    (255, 255, 0),
)

app = FastAPI(title="FoundationPose HTTP Service")
infer_lock = asyncio.Lock()

_fp_holder: Dict[str, Any] = {
    "est": None,
    "mesh_path": None,
    "bbox": None,
    "to_origin_inv": None,
}


class InferTiming(TypedDict, total=False):
    sam3_s: float
    pose_s: float
    vis_s: float
    pipeline_s: float
    upload_s: float
    total_s: float


def _output_root() -> Path:
    return Path(os.environ.get("FOUNDATIONPOSE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser().resolve()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    else:
        path = path.resolve()
    return path


def _mesh_path() -> Path:
    value = os.environ.get("FOUNDATIONPOSE_MESH_FILE", str(DEFAULT_MESH_FILE))
    path = _resolve_repo_path(value)
    if not path.is_file():
        raise RuntimeError(f"CAD mesh not found: {path} (FOUNDATIONPOSE_MESH_FILE={value!r})")
    return path


def _est_refine_iter() -> int:
    return int(os.environ.get("FOUNDATIONPOSE_EST_REFINE_ITER", "5"))


def _mesh_scale() -> float:
    """CAD 顶点单位到米的缩放，例如 mm 模型设 ``0.001``。"""
    return float(os.environ.get("FOUNDATIONPOSE_MESH_SCALE", str(DEFAULT_MESH_SCALE)))


async def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _camera_json_to_K(camera: Dict[str, Any]) -> np.ndarray:
    if "cam_K" not in camera:
        raise ValueError("camera.json must contain cam_K (9 floats, row-major 3x3)")
    k = camera["cam_K"]
    if len(k) != 9:
        raise ValueError("cam_K must have 9 elements (3x3 row-major)")
    return np.asarray(k, dtype=np.float64).reshape(3, 3)


def _load_rgb_array(rgb_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read rgb: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_depth_array(depth_path: Path, depth_scale: float) -> np.ndarray:
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"cannot read depth: {depth_path}")
    if d.ndim == 3:
        d = d[:, :, 0]
    d = np.asarray(d)
    if d.dtype == np.uint16:
        return d.astype(np.float32) * float(depth_scale)
    return d.astype(np.float32) * float(depth_scale)


def _prepare_depth_meters(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32, copy=True)
    depth[(depth < 0.001) | ~np.isfinite(depth)] = 0.0
    return depth


def _rotation_matrix_to_euler_zyx(rotation: List[List[float]]) -> List[float]:
    r00, r10 = float(rotation[0][0]), float(rotation[1][0])
    r20, r21, r22 = float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2])
    r01, r11 = float(rotation[0][1]), float(rotation[1][1])

    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy > 1e-6:
        rx = math.atan2(r21, r22)
        ry = math.atan2(-r20, sy)
        rz = math.atan2(r10, r00)
    else:
        rx = math.atan2(-r01, r11)
        ry = math.atan2(-r20, sy)
        rz = 0.0
    return [rx, ry, rz]


def _pose_4x4_to_t_and_R(pose_4x4: np.ndarray) -> tuple[List[float], List[List[float]]]:
    t_m = pose_4x4[:3, 3].astype(float).tolist()
    rot = pose_4x4[:3, :3].astype(float).tolist()
    return t_m, rot


def _visualize_poses(
    rgb: np.ndarray,
    K: np.ndarray,
    poses: List[np.ndarray],
    instance_ids: List[int],
    *,
    bbox: np.ndarray,
    to_origin_inv: np.ndarray,
) -> np.ndarray:
    from Utils import draw_posed_3d_box, draw_xyz_axis

    vis = rgb.copy()
    axis_scale = float(os.environ.get("FOUNDATIONPOSE_AXIS_SCALE", "0.1"))
    for idx, (inst_id, pose) in enumerate(zip(instance_ids, poses)):
        color = _POSE_COLORS_RGB[idx % len(_POSE_COLORS_RGB)]
        center_pose = pose @ to_origin_inv
        vis = draw_posed_3d_box(
            K,
            vis,
            ob_in_cam=center_pose,
            bbox=bbox,
            line_color=color,
            linewidth=2,
        )
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=axis_scale,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        # 在 mask 区域附近标注 instance id（简单放在图像顶部）
        cv2.putText(
            vis,
            f"id={inst_id}",
            (10, 28 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (color[2], color[1], color[0]),
            2,
            cv2.LINE_AA,
        )
    return vis


def _load_foundationpose_models(mesh_path: Path) -> None:
    import nvdiffrast.torch as dr

    from Utils import set_logging_format, set_seed
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(str(mesh_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    scale = _mesh_scale()
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(scale)
        print(f"[http_server] mesh scaled by {scale} (vertices now in meters)")

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=str(_output_root() / "_fp_debug"),
    )

    _fp_holder["est"] = est
    _fp_holder["mesh_path"] = str(mesh_path)
    _fp_holder["bbox"] = bbox
    _fp_holder["to_origin_inv"] = np.linalg.inv(to_origin)


def _get_estimator():
    est = _fp_holder.get("est")
    if est is None:
        raise RuntimeError("FoundationPose not loaded; check server startup logs")
    return est


def _run_foundationpose_pipeline(
    rgb_path: Path,
    depth_path: Path,
    camera_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    timing: InferTiming = {}

    with camera_path.open("r", encoding="utf-8") as f:
        camera_json = json.load(f)

    rgb = _load_rgb_array(rgb_path)
    h, w = rgb.shape[:2]
    K = _camera_json_to_K(camera_json)

    depth_scale = float(camera_json.get("depth_scale", 0.001))
    if depth_path.suffix.lower() == ".png" and depth_scale == 1.0:
        depth_scale = 0.001
    depth = _load_depth_array(depth_path, depth_scale)
    depth = _prepare_depth_meters(depth)

    if depth.shape[:2] != (h, w):
        raise ValueError(f"rgb/depth size mismatch: rgb={(h, w)}, depth={depth.shape[:2]}")

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    mask_path = results_dir / "mask_instances.png"
    detection_ism_path = results_dir / "detection_ism.json"
    detection_pose_path = results_dir / "detection_pose.json"
    vis_ism_path = results_dir / "vis_ism.png"
    vis_pose_path = results_dir / "vis_pose.png"
    vis_sam3_seg_path = results_dir / "vis_sam3_seg.png"

    sam3_max_inst = int(os.environ.get("GENPOSE2_SAM3_MAX_INSTANCES", "0"))
    prompt = os.environ.get("GENPOSE2_SAM3_PROMPT") or os.environ.get("SAM6D_SAM3_PROMPT")
    threshold = os.environ.get("GENPOSE2_SAM3_THRESHOLD") or os.environ.get("SAM6D_SAM3_THRESHOLD")
    mask_threshold = os.environ.get("GENPOSE2_SAM3_MASK_THRESHOLD") or os.environ.get(
        "SAM6D_SAM3_MASK_THRESHOLD"
    )

    t0 = time.perf_counter()
    sam3_result: Sam3SegmentationResult = run_sam3_segmentation(
        rgb_path,
        output_dir,
        prompt=prompt,
        threshold=float(threshold) if threshold is not None else None,
        mask_threshold=float(mask_threshold) if mask_threshold is not None else None,
        mask_exr_out=mask_path,
        max_instances=sam3_max_inst,
    )
    timing["sam3_s"] = time.perf_counter() - t0

    if detection_ism_path.is_file():
        pass
    elif sam3_result.detection_ism_path.is_file():
        detection_ism_path.write_text(
            sam3_result.detection_ism_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    sam6d_ism = output_dir / "sam6d_results" / "detection_ism.json"
    if sam6d_ism.is_file() and not detection_ism_path.is_file():
        detection_ism_path.write_text(sam6d_ism.read_text(encoding="utf-8"), encoding="utf-8")

    if sam3_result.vis_ism_path and sam3_result.vis_ism_path.is_file():
        vis_ism_path.write_bytes(sam3_result.vis_ism_path.read_bytes())
    elif sam3_result.instance_dets:
        visualize_sam3_ism(
            rgb_path,
            sam3_result.instance_dets,
            vis_ism_path,
            prompt=prompt,
            instance_ids=list(range(1, sam3_result.num_instances + 1)),
        )

    if mask_path.is_file() or sam3_result.mask_exr.is_file():
        src_mask = mask_path if mask_path.is_file() else sam3_result.mask_exr
        visualize_sam3_mask_exr(rgb_path, src_mask, vis_sam3_seg_path)

    if not sam3_result.instance_dets:
        raise RuntimeError("SAM3 returned no instances")

    est = _get_estimator()
    bbox = _fp_holder["bbox"]
    to_origin_inv = _fp_holder["to_origin_inv"]
    refine_iter = _est_refine_iter()

    detections: List[Dict[str, Any]] = []
    pose_mats: List[np.ndarray] = []
    pose_instance_ids: List[int] = []

    t0 = time.perf_counter()
    for idx, det in enumerate(sam3_result.instance_dets):
        inst_id = idx + 1
        ob_mask = get_instance_bool_masks([det], (w, h))[0]
        if int(ob_mask.sum()) < 4:
            print(f"[http_server] skip instance id={inst_id}: mask too small")
            continue

        depth_i = depth.copy()
        pose = est.register(
            K=K,
            rgb=rgb,
            depth=depth_i,
            ob_mask=ob_mask.astype(bool),
            iteration=refine_iter,
        )
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        t_m, rot = _pose_4x4_to_t_and_R(pose)
        t_mm = (np.asarray(t_m) * 1000.0).tolist()
        euler_zyx_rad = _rotation_matrix_to_euler_zyx(rot)
        fp_score = None
        if getattr(est, "scores", None) is not None and len(est.scores) > 0:
            fp_score = float(est.scores[0].detach().cpu().item())

        entry: Dict[str, Any] = {
            "instance_id": inst_id,
            "seg_score": float(sam3_result.instance_scores[idx]) if sam3_result.instance_scores else float(det.get("score", 0.0)),
            "pose_score": fp_score,
            "bbox": det.get("bbox"),
            "pose_4x4": pose.reshape(4, 4).tolist(),
            "t_m": t_m,
            "t_mm": t_mm,
            "R": rot,
            "rotation_euler_zyx_rad": euler_zyx_rad,
            "xyzrxryrz": list(t_mm) + list(euler_zyx_rad),
        }
        detections.append(entry)
        pose_mats.append(pose)
        pose_instance_ids.append(inst_id)

        pose_txt = results_dir / f"pose_inst{inst_id:02d}.txt"
        np.savetxt(str(pose_txt), pose.reshape(4, 4))
        entry["pose_path"] = str(pose_txt)

    timing["pose_s"] = time.perf_counter() - t0

    if not detections:
        raise RuntimeError("FoundationPose returned no valid poses (all masks too small?)")

    t0 = time.perf_counter()
    vis_pose_rgb = _visualize_poses(
        rgb,
        K,
        pose_mats,
        pose_instance_ids,
        bbox=bbox,
        to_origin_inv=to_origin_inv,
    )
    cv2.imwrite(str(vis_pose_path), cv2.cvtColor(vis_pose_rgb, cv2.COLOR_RGB2BGR))
    timing["vis_s"] = time.perf_counter() - t0

    detection_pose_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")

    best = detections[0]
    timing["pipeline_s"] = float(timing.get("sam3_s", 0.0)) + float(timing.get("pose_s", 0.0)) + float(
        timing.get("vis_s", 0.0)
    )

    payload: Dict[str, Any] = {
        "num_instances": len(detections),
        "seg_num_instances": sam3_result.num_instances,
        "score": float(best["seg_score"]),
        "pose_score": best.get("pose_score"),
        "xyz_mm": best["t_mm"],
        "rotation_euler_zyx_rad": best["rotation_euler_zyx_rad"],
        "rotation_order": "zyx",
        "pose_convention": "xyz is camera-frame translation in mm; rx, ry, rz are ZYX Euler angles in radians.",
        "xyzrxryrz": best["xyzrxryrz"],
        "xyzrxryrz_unit": "mm_rad",
        "pose_4x4": best["pose_4x4"],
        "mesh_file": _fp_holder.get("mesh_path"),
        "result_dir": str(output_dir),
        "detection_ism_path": str(detection_ism_path if detection_ism_path.is_file() else sam3_result.detection_ism_path),
        "detection_pose_path": str(detection_pose_path),
        "vis_ism_path": str(vis_ism_path),
        "vis_pose_path": str(vis_pose_path),
        "vis_sam3_seg_path": str(vis_sam3_seg_path) if vis_sam3_seg_path.is_file() else None,
        "detections": detections,
        "timing": timing,
        "sam3_prompt": prompt or os.environ.get("GENPOSE2_SAM3_PROMPT", DEFAULT_SAM3_PROMPT),
    }
    return payload


@app.on_event("startup")
async def _startup_load_models() -> None:
    py = _sam3_python()
    script = _sam3_infer_script()
    print(
        f"[http_server] SAM3: root={_sam3_root()} python={py} infer_script={script} "
        f"(default infer: {DEFAULT_SAM3_INFER_SCRIPT})"
    )
    try:
        _validate_sam3_toolchain(py, script)
        print("[http_server] SAM3 toolchain: ok")
    except FileNotFoundError as exc:
        print(f"[http_server] SAM3 toolchain missing: {exc}")

    try:
        from sam3_seg import _cocomask

        _cocomask()
        print("[http_server] SAM3 dependency pycocotools: ok")
    except ImportError as exc:
        print(f"[http_server] SAM3 dependency missing: {exc}")

    try:
        mesh_path = _mesh_path()
        print(f"[http_server] loading FoundationPose mesh={mesh_path} ...")
        _load_foundationpose_models(mesh_path)
        print("[http_server] FoundationPose models loaded")
    except Exception as exc:
        print(f"[http_server] FoundationPose load failed: {type(exc).__name__}: {exc}")


@app.get("/health")
def health() -> Dict[str, Any]:
    mesh_cfg = os.environ.get("FOUNDATIONPOSE_MESH_FILE", str(DEFAULT_MESH_FILE))
    try:
        mesh_resolved = str(_mesh_path())
        mesh_exists = True
    except Exception:
        mesh_resolved = str(_resolve_repo_path(mesh_cfg))
        mesh_exists = Path(mesh_resolved).is_file()

    return {
        "status": "ok",
        "root_dir": str(ROOT_DIR),
        "output_root": str(_output_root()),
        "foundationpose_loaded": _fp_holder.get("est") is not None,
        "mesh_file": mesh_cfg,
        "mesh_file_resolved": mesh_resolved,
        "mesh_file_exists": mesh_exists,
        "mesh_scale": _mesh_scale(),
        "est_refine_iter": _est_refine_iter(),
        "sam3_root": str(_sam3_root()),
        "sam3_python": _sam3_python(),
        "sam3_python_default": DEFAULT_SAM3_PYTHON,
        "sam3_infer_script": str(_sam3_infer_script()),
        "sam3_infer_script_default": DEFAULT_SAM3_INFER_SCRIPT,
        "sam3_prompt": os.environ.get("GENPOSE2_SAM3_PROMPT")
        or os.environ.get("SAM6D_SAM3_PROMPT", DEFAULT_SAM3_PROMPT),
    }


@app.post("/infer")
async def infer(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    camera: UploadFile = File(...),
) -> JSONResponse:
    if _fp_holder.get("est") is None:
        raise HTTPException(status_code=503, detail="FoundationPose not loaded on server startup")

    request_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_dir = _output_root() / request_id
    input_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=False)

    rgb_path = input_dir / "rgb.png"
    depth_path = input_dir / "depth.png"
    camera_path = input_dir / "camera.json"

    t0 = time.perf_counter()
    await _save_upload(rgb, rgb_path)
    await _save_upload(depth, depth_path)
    await _save_upload(camera, camera_path)
    upload_s = time.perf_counter() - t0

    try:
        with camera_path.open("r", encoding="utf-8") as f:
            camera_json = json.load(f)
        if "cam_K" not in camera_json:
            raise ValueError("camera.json must contain cam_K")
        if "depth_scale" not in camera_json:
            raise ValueError("camera.json must contain depth_scale")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid camera.json: {exc}") from exc

    async with infer_lock:
        try:
            payload = await asyncio.to_thread(
                _run_foundationpose_pipeline,
                rgb_path,
                depth_path,
                camera_path,
                output_dir,
            )
            timing = payload["timing"]
            timing["upload_s"] = upload_s
            timing["total_s"] = upload_s + float(timing["pipeline_s"])
            payload["timing"] = timing
            print(
                f"[http_server] infer {request_id} ok "
                f"instances={payload.get('num_instances')} total={timing['total_s']:.3f}s"
            )
            return JSONResponse(content=payload)
        except HTTPException:
            raise
        except Exception as exc:
            print(f"[http_server] infer {request_id} failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundationPose HTTP service (SAM3 + register)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8002, type=int)
    parser.add_argument(
        "--mesh-file",
        default=os.environ.get("FOUNDATIONPOSE_MESH_FILE", str(DEFAULT_MESH_FILE)),
        help="CAD mesh path for FoundationPose",
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=float(os.environ.get("FOUNDATIONPOSE_MESH_SCALE", str(DEFAULT_MESH_SCALE))),
        help="Scale CAD vertices to meters (e.g. 0.001 for mm meshes)",
    )
    args = parser.parse_args()
    os.environ["FOUNDATIONPOSE_MESH_FILE"] = args.mesh_file
    os.environ["FOUNDATIONPOSE_MESH_SCALE"] = str(args.mesh_scale)
    if not os.environ.get("GENPOSE2_SAM3_PROMPT") and not os.environ.get("SAM6D_SAM3_PROMPT"):
        os.environ["GENPOSE2_SAM3_PROMPT"] = DEFAULT_SAM3_PROMPT

    import uvicorn

    print(
        f"[http_server] starting uvicorn mesh={args.mesh_file} "
        f"mesh_scale={args.mesh_scale} sam3_prompt={os.environ.get('GENPOSE2_SAM3_PROMPT')}"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
