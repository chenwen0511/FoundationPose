"""
VLM ROI + 全尺寸白底图 + 实例分割（YOLO 或 SAM3）。

流程见 ``seg/vlm_seg.md``：
  ① VLM bbox → ② ROI 外置白 ``new_image`` → ③ ``run_yolo_segmentation_ism`` / ``run_sam3_segmentation``

默认（启用 VLM 时）：``GENPOSE2_SEG_BACKEND=yolo``；未启用 VLM 时默认 ``sam3``。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 支持 ``python seg/vlm_seg.py`` 与 ``python -m seg.vlm_seg``（需将仓库根目录加入 path）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from seg.sam3_seg import (
    DEFAULT_VLM_ROI_MARGIN_PX,
    Sam3SegmentationResult,
    run_sam3_segmentation,
)

DEFAULT_VLM_API_URL = "http://192.168.100.92:8000/v1/chat/completions"
DEFAULT_VLM_MODEL = "qwen3-vl-4b"
DEFAULT_VLM_TEMPERATURE = 0.2
DEFAULT_VLM_TIMEOUT_S = 120.0
DEFAULT_VLM_ROI_MIN_AREA_PX = 100
DEFAULT_VLM_PROMPT = """
        Detect the single white plastic tray / plate / circular material tray that is directly above the blue dot marker.

        Important visual rule:
        - First locate the blue circular dot marker.
        - Then find the ONE white vertical tray / plate / circular plastic piece whose horizontal position is aligned with the blue dot.
        - The target is the white tray immediately above the blue dot, not the blue dot itself.
        - The bbox must cover only one narrow white vertical tray aligned with the blue dot, never the full stack.
        - Do NOT detect other white trays on the left or right.
        - Do NOT detect the metal rack, background, or blue dot.

        Output format:
        [
          {"bbox_2d": [x1, y1, x2, y2], "label": "white_tray_above_blue_dot"}
        ]

        bbox rule:
        - bbox_2d must tightly cover only the visible target white tray / plate above the blue dot.
        - Coordinates must be relative coordinates from 0 to 1000.
        - x1 < x2 and y1 < y2.

        Strict rules:
        - Return at most one bbox.
        - If the blue dot is not visible, return [].
        - If the white tray directly above the blue dot is not clear, return [].
        - label must be exactly "white_tray_above_blue_dot".
        - Return JSON array only.
        - No explanation.
        - No markdown.
        """.strip()

MASKED_RGB_NAME = "rgb_vlm_masked.png"


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def use_vlm_roi() -> bool:
    return _env_first("GENPOSE2_USE_VLM_ROI", default="1").lower() not in ("0", "false", "no")


def seg_backend() -> str:
    """
    ``GENPOSE2_SEG_BACKEND=yolo|sam3``。
    未设置时：启用 VLM → ``yolo``；否则 ``sam3``。
    """
    explicit = _env_first("GENPOSE2_SEG_BACKEND", default="").lower()
    if explicit:
        if explicit not in ("yolo", "sam3"):
            raise ValueError(f"unsupported GENPOSE2_SEG_BACKEND={explicit!r}, use yolo or sam3")
        return explicit
    return "yolo" if use_vlm_roi() else "sam3"


def _run_instance_segmentation(
    image_path: Path,
    output_dir: Path,
    *,
    original_rgb_path: Optional[Path] = None,
    sam3_prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    mask_exr_out: Optional[Path] = None,
    max_instances: int = 0,
) -> Sam3SegmentationResult:
    """在 ``image_path``（通常为 VLM 后的 new_image）上跑 YOLO 或 SAM3。"""
    backend = seg_backend()
    print(f"[vlm_seg] instance segmentation backend={backend} image={image_path}")
    if backend == "yolo":
        from seg.yolo_seg import run_yolo_segmentation_ism

        max_inst = max_instances if max_instances > 0 else 1
        return run_yolo_segmentation_ism(
            image_path,
            output_dir,
            max_instances=max_inst,
            mask_exr_out=mask_exr_out,
            original_rgb_path=original_rgb_path,
        )
    return run_sam3_segmentation(
        image_path,
        output_dir,
        prompt=sam3_prompt,
        threshold=threshold,
        mask_threshold=mask_threshold,
        mask_exr_out=mask_exr_out,
        max_instances=max_instances,
    )


def _vlm_api_url() -> str:
    return _env_first("GENPOSE2_VLM_API_URL", default=DEFAULT_VLM_API_URL)


def _vlm_model() -> str:
    return _env_first("GENPOSE2_VLM_MODEL", default=DEFAULT_VLM_MODEL)


def _vlm_prompt() -> str:
    return _env_first("GENPOSE2_VLM_PROMPT", default=DEFAULT_VLM_PROMPT)


def _vlm_roi_margin_px() -> int:
    return max(0, int(_env_first(
        "GENPOSE2_VLM_ROI_MARGIN_PX",
        default=str(DEFAULT_VLM_ROI_MARGIN_PX),
    )))


def _vlm_roi_fill_rgb() -> Tuple[int, int, int]:
    """
    ROI 外填充色。``GENPOSE2_VLM_ROI_FILL=white|black``（默认 white）。
    示例图常用黑色背景，与白色效果相同：均为抹掉 ROI 外其它托盘干扰。
    """
    fill = _env_first("GENPOSE2_VLM_ROI_FILL", default="white").lower()
    if fill in ("black", "0", "#000", "#000000"):
        return (0, 0, 0)
    if fill in ("white", "255", "#fff", "#ffffff"):
        return (255, 255, 255)
    raise ValueError(f"unsupported GENPOSE2_VLM_ROI_FILL={fill!r}, use white or black")


def _vlm_roi_min_area() -> int:
    return max(1, int(_env_first("GENPOSE2_VLM_ROI_MIN_AREA_PX", default=str(DEFAULT_VLM_ROI_MIN_AREA_PX))))


def clip_int(v: float, low: int, high: int) -> int:
    return int(np.clip(round(v), low, high))


def bbox_to_pixel(
    bbox: List[float],
    W: int,
    H: int,
    mode: str = "norm1000",
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if mode == "pixel":
        px1 = clip_int(x1, 0, W - 1)
        py1 = clip_int(y1, 0, H - 1)
        px2 = clip_int(x2, 0, W - 1)
        py2 = clip_int(y2, 0, H - 1)
    elif mode == "norm1000":
        px1 = clip_int(x1 / 1000.0 * (W - 1), 0, W - 1)
        py1 = clip_int(y1 / 1000.0 * (H - 1), 0, H - 1)
        px2 = clip_int(x2 / 1000.0 * (W - 1), 0, W - 1)
        py2 = clip_int(y2 / 1000.0 * (H - 1), 0, H - 1)
    else:
        if max(x1, y1, x2, y2) <= 1000.0 and (W > 1100 or H > 1100):
            px1 = clip_int(x1 / 1000.0 * (W - 1), 0, W - 1)
            py1 = clip_int(y1 / 1000.0 * (H - 1), 0, H - 1)
            px2 = clip_int(x2 / 1000.0 * (W - 1), 0, W - 1)
            py2 = clip_int(y2 / 1000.0 * (H - 1), 0, H - 1)
        else:
            px1 = clip_int(x1, 0, W - 1)
            py1 = clip_int(y1, 0, H - 1)
            px2 = clip_int(x2, 0, W - 1)
            py2 = clip_int(y2, 0, H - 1)
    xx1, xx2 = sorted([px1, px2])
    yy1, yy2 = sorted([py1, py2])
    return xx1, yy1, xx2, yy2


def _image_to_base64(file_path: Path) -> str:
    import base64

    with file_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_vlm_detections(generated_text: str) -> Optional[Dict[str, Any]]:
    text = _strip_json_fence(generated_text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    det = data[0]
    if not isinstance(det, dict) or "bbox_2d" not in det:
        return None
    bbox = det["bbox_2d"]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    return det


@dataclass
class VlmRoiResult:
    bbox_pixel: Tuple[int, int, int, int]
    label: str
    raw_response: str
    bbox_norm: Optional[List[float]] = None


@dataclass
class VlmSam3SegmentationResult:
    sam3: Sam3SegmentationResult
    vlm_used: bool
    vlm_bbox: Optional[Tuple[int, int, int, int]] = None
    vlm_label: Optional[str] = None
    masked_rgb_path: Optional[Path] = None
    vlm_roi_json_path: Optional[Path] = None
    vlm_fallback: bool = False
    vlm_fallback_reason: Optional[str] = None
    timing: Dict[str, float] = field(default_factory=dict)


def detect_vlm_roi(
    rgb_path: Path,
    *,
    prompt: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_VLM_TIMEOUT_S,
    temperature: Optional[float] = None,
) -> Optional[VlmRoiResult]:
    """调用 VLM 返回单个 ROI；无目标或解析失败返回 None。"""
    rgb_path = rgb_path.expanduser().resolve()
    if not rgb_path.is_file():
        raise FileNotFoundError(f"rgb not found: {rgb_path}")

    img = Image.open(rgb_path).convert("RGB")
    W, H = img.size
    prompt_text = prompt if prompt is not None else _vlm_prompt()
    url = api_url if api_url is not None else _vlm_api_url()
    model_name = model if model is not None else _vlm_model()
    temp = DEFAULT_VLM_TEMPERATURE if temperature is None else float(temperature)

    image_base64 = _image_to_base64(rgb_path)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": temp,
    }

    print(f"[vlm_seg] POST {url} model={model_name!r}")
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout_s)
    except requests.RequestException as exc:
        raise RuntimeError(f"VLM 服务不可达: {url}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"VLM request failed: status={resp.status_code} body={resp.text[:500]}")

    generated_text = resp.json()["choices"][0]["message"]["content"]
    det = _parse_vlm_detections(generated_text)
    if det is None:
        print(f"[vlm_seg] VLM parse failed or empty: {generated_text[:200]!r}")
        return None

    bbox_norm = [float(x) for x in det["bbox_2d"]]
    x1, y1, x2, y2 = bbox_to_pixel(bbox_norm, W, H)
    if (x2 - x1 + 1) * (y2 - y1 + 1) < _vlm_roi_min_area():
        print(f"[vlm_seg] ROI too small: {(x1, y1, x2, y2)}")
        return None

    label = str(det.get("label", "roi"))
    print(f"[vlm_seg] ROI pixel=({x1},{y1},{x2},{y2}) label={label!r}")
    return VlmRoiResult(
        bbox_pixel=(x1, y1, x2, y2),
        label=label,
        raw_response=generated_text,
        bbox_norm=bbox_norm,
    )


def _expand_bbox(
    bbox: Tuple[int, int, int, int],
    margin_px: int,
    W: int,
    H: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = clip_int(x1 - margin_px, 0, W - 1)
    y1 = clip_int(y1 - margin_px, 0, H - 1)
    x2 = clip_int(x2 + margin_px, 0, W - 1)
    y2 = clip_int(y2 + margin_px, 0, H - 1)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def build_roi_masked_image(
    rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    fill_value: Optional[Tuple[int, int, int]] = None,
    margin_px: int = 0,
) -> np.ndarray:
    """ROI 外填纯色（白/黑），ROI 内保留原像素；输出 ``new_image`` 与 ``rgb`` 同 shape。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    if fill_value is None:
        fill_value = _vlm_roi_fill_rgb()
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(bbox, margin_px, W, H)
    out = np.full_like(rgb, fill_value, dtype=rgb.dtype)
    out[y1 : y2 + 1, x1 : x2 + 1] = rgb[y1 : y2 + 1, x1 : x2 + 1]
    assert out.shape == rgb.shape
    return out


# 兼容旧名
build_roi_whited_image = build_roi_masked_image


def load_rgb_from_path(rgb_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {rgb_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb_png(rgb: np.ndarray, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def _image_hw(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        w, h = img.size
    return h, w


def write_new_image_for_sam3(
    rgb_path: Path,
    roi: VlmRoiResult,
    output_dir: Path,
    *,
    margin_px: Optional[int] = None,
) -> Tuple[Path, np.ndarray]:
    """
    生成并落盘 ``new_image``：与原 rgb **同 H×W**，ROI 内保留原像素，ROI 外为白色。

    返回 ``(rgb_vlm_masked.png 路径, new_image 数组)``，供 SAM3 ``--image`` 唯一使用。
    """
    rgb_path = rgb_path.expanduser().resolve()
    rgb = load_rgb_from_path(rgb_path)
    H, W = rgb.shape[:2]
    margin = _vlm_roi_margin_px() if margin_px is None else margin_px
    fill = _vlm_roi_fill_rgb()
    new_image = build_roi_masked_image(rgb, roi.bbox_pixel, fill_value=fill, margin_px=margin)
    if new_image.shape != rgb.shape:
        raise RuntimeError(
            f"new_image shape {new_image.shape} != original rgb {rgb.shape}; "
            "禁止 resize/crop"
        )

    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    masked_path = input_dir / MASKED_RGB_NAME
    save_rgb_png(new_image, masked_path)
    # 便于直接打开查看（与示例图一致：中间竖条 ROI，两侧纯色）
    preview_path = output_dir / MASKED_RGB_NAME
    save_rgb_png(new_image, preview_path)

    mh, mw = _image_hw(masked_path)
    if (mh, mw) != (H, W):
        raise RuntimeError(
            f"saved new_image size ({mw}x{mh}) != original rgb ({W}x{H})"
        )
    fill_name = "black" if fill == (0, 0, 0) else "white"
    print(
        f"[vlm_seg] new_image -> {masked_path} "
        f"(preview {preview_path}) "
        f"size={W}x{H} same as rgb, ROI={roi.bbox_pixel}, outside={fill_name}"
    )
    return masked_path, new_image


def draw_vlm_roi_vis(
    rgb_path: Path,
    bbox: Tuple[int, int, int, int],
    output_path: Path,
    *,
    label: str = "",
) -> Path:
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    x1, y1, x2, y2 = bbox
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
    if label:
        draw.text((x1, max(0, y1 - 22)), label, fill="red", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _write_vlm_roi_json(
    path: Path,
    *,
    roi: Optional[VlmRoiResult],
    masked_rgb: Optional[Path],
    fallback: bool,
    fallback_reason: Optional[str],
    sam3_image: Optional[Path] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "vlm_used": roi is not None,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "masked_rgb": str(masked_rgb) if masked_rgb else None,
        "sam3_image": str(sam3_image) if sam3_image else None,
        "sam3_uses_new_image": roi is not None and not fallback,
        "roi_fill": _env_first("GENPOSE2_VLM_ROI_FILL", default="white"),
    }
    if roi is not None:
        payload.update(
            {
                "bbox_pixel": list(roi.bbox_pixel),
                "bbox_norm": roi.bbox_norm,
                "label": roi.label,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_vlm_sam3_segmentation(
    rgb_path: Path,
    output_dir: Path,
    *,
    vlm_prompt: Optional[str] = None,
    sam3_prompt: Optional[str] = None,
    threshold: Optional[float] = None,
    mask_threshold: Optional[float] = None,
    mask_exr_out: Optional[Path] = None,
    max_instances: int = 0,
    skip_vlm: bool = False,
) -> VlmSam3SegmentationResult:
    """
    ① VLM ROI → ② 白底 ``new_image`` → ③ YOLO/SAM3（``GENPOSE2_SEG_BACKEND``，VLM 开启时默认 yolo）。

    **启用 VLM 时实例分割的输入必须是 ``new_image``。VLM 失败时直接报错。**
    """
    rgb_path = rgb_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timing: Dict[str, float] = {}
    roi: Optional[VlmRoiResult] = None
    masked_path: Optional[Path] = None
    vlm_enabled = not skip_vlm and use_vlm_roi()
    seg_image_path: Path = rgb_path
    backend = seg_backend()

    if vlm_enabled:
        t0 = time.perf_counter()
        try:
            roi = detect_vlm_roi(rgb_path, prompt=vlm_prompt)
        except Exception as exc:
            raise RuntimeError(
                f"VLM 调用失败，已中止（请检查 {_vlm_api_url()} 是否可访问）: {exc}"
            ) from exc
        timing["vlm_s"] = time.perf_counter() - t0

        if roi is None:
            raise RuntimeError(
                "VLM 未返回有效 ROI（空框或 JSON 解析失败），无法生成 new_image，已中止。"
            )

        t0 = time.perf_counter()
        masked_path, _new_image = write_new_image_for_sam3(rgb_path, roi, output_dir)
        seg_image_path = masked_path
        timing["vlm_mask_s"] = time.perf_counter() - t0
        draw_vlm_roi_vis(
            rgb_path,
            roi.bbox_pixel,
            results_dir / "vlm_roi_vis.png",
            label=roi.label,
        )
    else:
        print(f"[vlm_seg] VLM 未启用，{backend} 使用原图 rgb")

    if vlm_enabled and roi is not None:
        if seg_image_path.resolve() != masked_path.resolve():
            raise RuntimeError(
                f"VLM 已启用且 ROI 有效，但分割输入不是 new_image: {seg_image_path}"
            )
        fill_name = _env_first("GENPOSE2_VLM_ROI_FILL", default="white")
        print(
            f"[vlm_seg] {backend} input={seg_image_path} "
            f"(new_image: ROI 内原图 + ROI 外 {fill_name}, 与原 rgb 同尺寸)"
        )

    t0 = time.perf_counter()
    sam3_result = _run_instance_segmentation(
        seg_image_path,
        output_dir,
        original_rgb_path=rgb_path if (vlm_enabled and roi is not None) else None,
        sam3_prompt=sam3_prompt,
        threshold=threshold,
        mask_threshold=mask_threshold,
        mask_exr_out=mask_exr_out,
        max_instances=max_instances,
    )
    timing["seg_s"] = time.perf_counter() - t0
    timing[f"{backend}_s"] = timing["seg_s"]

    roi_json_path = output_dir / "vlm_roi.json"
    for path in (results_dir / "vlm_roi.json", roi_json_path):
        _write_vlm_roi_json(
            path,
            roi=roi,
            masked_rgb=masked_path,
            fallback=False,
            fallback_reason=None,
            sam3_image=seg_image_path,
        )
    meta_path = output_dir / "vlm_roi.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["seg_backend"] = backend
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return VlmSam3SegmentationResult(
        sam3=sam3_result,
        vlm_used=roi is not None,
        vlm_bbox=roi.bbox_pixel if roi else None,
        vlm_label=roi.label if roi else None,
        masked_rgb_path=masked_path,
        vlm_roi_json_path=roi_json_path,
        vlm_fallback=False,
        vlm_fallback_reason=None,
        timing=timing,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VLM ROI + white mask + YOLO/SAM3 on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument(
        "--seg-backend",
        choices=("yolo", "sam3"),
        default=None,
        help="override GENPOSE2_SEG_BACKEND",
    )
    args = parser.parse_args()
    if args.seg_backend:
        os.environ["GENPOSE2_SEG_BACKEND"] = args.seg_backend

    out = run_vlm_sam3_segmentation(
        args.image,
        args.output_dir,
        skip_vlm=args.skip_vlm,
    )
    vis = {}
    if out.sam3.vis_ism_path:
        vis["vis_ism"] = str(out.sam3.vis_ism_path)
    results_dir = args.output_dir / "results"
    for name in ("vis_ism.png", "vis_sam3_seg.png", "vis_ism_orig.png", "vis_seg_orig.png", "detection_ism.json", "mask_instances.png"):
        p = results_dir / name
        if p.is_file():
            vis[name] = str(p.resolve())
    print(json.dumps(
        {
            "vlm_used": out.vlm_used,
            "vlm_bbox": out.vlm_bbox,
            "masked_rgb": str(out.masked_rgb_path),
            "detection_ism": str(out.sam3.detection_ism_path),
            "results": vis,
            "timing": out.timing,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
