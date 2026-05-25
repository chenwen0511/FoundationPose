# 测试样例 — Plastic Reel

## 数据文件

```
test/
├── CAD/
│   └── tray_180mm_centered_mesh_v2.ply   # 托盘 CAD，顶点单位 mm
├── 20260509_145205_5a80ad10/inputs/      # 单实例样例
│   ├── rgb.png
│   ├── depth.png
│   └── camera.json
└── 20260522_121946/                    # 2026-05-22 采集样例（文件在目录根）
    ├── rgb.png
    ├── depth.png
    ├── camera.json
    └── capture_meta.json
```

## 关键配置

| 项 | 值 |
|----|-----|
| VLM+分割 | `seg/vlm_seg.py`（VLM ROI → 白底 `new_image` → **YOLO** 或 SAM3） |
| YOLO | `seg/yolo_seg.py`（权重 `GENPOSE2_YOLO_WEIGHTS`，默认 `seg/weights/yolo.pt`） |
| SAM3 封装 | `seg/sam3_seg.py`（子进程调用 `sam3/scripts/infer.py`） |
| CAD | `test/CAD/tray_180mm_centered_mesh_v2.ply` |
| CAD 缩放 | `FOUNDATIONPOSE_MESH_SCALE=0.001`（mm → m） |
| SAM3 提示词 | `"Plastic Reel"` |

## 启动服务

```bash
conda activate foundationpose

python http_server.py --host 0.0.0.0 --port 8002
# 默认已使用 test/CAD/tray_180mm_centered_mesh_v2.ply、mesh_scale=0.001、prompt="Plastic Reel"
```

或显式指定：

```bash
export FOUNDATIONPOSE_MESH_FILE=test/CAD/tray_180mm_centered_mesh_v2.ply
export FOUNDATIONPOSE_MESH_SCALE=0.001
export GENPOSE2_SAM3_PROMPT="Plastic Reel"
export GENPOSE2_SAM3_ROOT=/home/ubuntu/stephen/01-code/sam3
export GENPOSE2_VLM_API_URL=http://192.168.100.92:8000/v1/chat/completions
export GENPOSE2_USE_VLM_ROI=1
export GENPOSE2_SEG_BACKEND=yolo
export GENPOSE2_YOLO_WEIGHTS=seg/weights/yolo.pt
python http_server.py --port 8002
```

## 调用测试

```bash
python test/call_infer.py --sample-dir test/20260509_145205_5a80ad10/inputs
python test/run_local_infer.py --sample-dir test/20260509_145205_5a80ad10/inputs
python test/run_local_infer.py --sample-dir test/20260522_121946 --no-vlm
bash test/run_test.sh

# 20260522_121946 专用脚本（sample 在目录根，无需 inputs/）
bash test/run_20260522_121946.sh              # 离线推理
bash test/run_20260522_121946.sh http         # HTTP 推理（需先启动 http_server）
python test/run_20260522_121946.py --check-only
python test/run_20260522_121946.py            # 等价于 bash 脚本 local 模式
```

## 输出

`service_outputs/<request_id>/results/`：

- `vlm_roi.json` / `vlm_roi_vis.png` — VLM ROI（启用时）
- `inputs/rgb_vlm_masked.png` — ROI 外白底、同尺寸 new_image
- `vis_ism.png` — SAM3 分割（Plastic Reel）
- `vis_sam3_seg.png` — SAM3 mask 叠加
- `vis_pose.png` — 3D 框 + 坐标轴
- `detection_pose.json` — 位姿列表
