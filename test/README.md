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
| VLM+分割 | `seg/vlm_seg.py`（原图 `SAM3 -> instances` + 原图 `VLM -> ROI` + 选取 ROI 交集最大的实例） |
| SAM3 封装 | `seg/sam3_seg.py`（REST 调用在线 SAM3 `/infer` 接口） |
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
export GENPOSE2_SAM3_API_URL=http://127.0.0.1:18002/infer
export GENPOSE2_VLM_API_URL=http://192.168.100.92:8000/v1/chat/completions
export GENPOSE2_USE_VLM_ROI_FILTER=1
export GENPOSE2_VLM_ROI_MARGIN_PX=10
export GENPOSE2_VLM_MIN_INTERSECTION_PX=1
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

- `vlm_roi.json` / `vlm_roi_vis.png` — VLM ROI 与交集筛选元数据（启用时）
- `detection_ism_raw.json` — SAM3 原始实例结果
- `detection_ism_filtered.json` / `detection_ism.json` — 仅保留与 ROI 交集最大的实例
- `vis_ism_raw.png` / `vis_ism.png` — 原始 / 筛选后 SAM3 实例可视化
- `vis_sam3_seg.png` — 筛选后实例 mask 叠加
- `vis_pose.png` — 3D 框 + 坐标轴
- `detection_pose.json` — 位姿列表

## http 调用
```
(base) ubuntu@ubuntu-System-Product-Name:~$ curl -X POST http://127.0.0.1:8002/infer   -F "rgb=@/home/ubuntu/stephen/01-code/FoundationPose/test/20260507_105248_1d1db1bb/inputs/rgb.png;type=image/png"   -F "depth=@/home/ubuntu/stephen/01-code/FoundationPose/test/20260507_105248_1d1db1bb/inputs/depth.png;type=image/png"   -F "camera=@/home/ubuntu/stephen/01-code/FoundationPose/test/20260507_105248_1d1db1bb/inputs/camera.json;type=application/json" | python -m "json.tool"
```

```

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  680k  100  3584  100  676k    390  75506  0:00:09  0:00:09 --:--:--   902
{
    "num_instances": 1,
    "seg_num_instances": 1,
    "seg_num_instances_raw": 6,
    "seg_score_min": 0.4,
    "pose_reproj_max_px": 40.0,
    "skipped_low_seg": [],
    "score": 0.796875,
    "pose_score": 105.66796875,
    "xyz_mm": [
        -53.77276950756606,
        2.931274253168315,
        323.0000138282776
    ],
    "rotation_euler_zyx_rad": [
        2.78175333027043,
        0.8686851987977124,
        1.1987914503922514
    ],
    "rotation_order": "zyx",
    "pose_convention": "xyz is camera-frame translation in mm; rx, ry, rz are ZYX Euler angles in radians.",
    "xyzrxryrz": [
        -53.77276950756606,
        2.931274253168315,
        323.0000138282776,
        2.78175333027043,
        0.8686851987977124,
        1.1987914503922514
    ],
    "xyzrxryrz_unit": "mm_rad",
    "pose_4x4": [
        [
            0.23474913835525513,
            0.9696536064147949,
            0.06829948723316193,
            -0.053772769507566064
        ],
        [
            0.6016563773155212,
            -0.08975300937891006,
            -0.793696403503418,
            0.002931274253168315
        ],
        [
            -0.7634804248809814,
            0.22741246223449707,
            -0.6044676303863525,
            0.3230000138282776
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ],
    "mesh_file": "/home/ubuntu/stephen/01-code/FoundationPose/test/CAD/tray_180mm_centered_mesh_v2.ply",
    "result_dir": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e",
    "detection_ism_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/detection_ism.json",
    "detection_pose_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/detection_pose.json",
    "vis_ism_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/vis_ism.png",
    "vis_pose_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/vis_pose.png",
    "vis_sam3_seg_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/vis_sam3_seg.png",
    "vlm_roi_json_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/vlm_roi.json",
    "vlm": {
        "use_vlm_roi_filter": true,
        "pipeline": "sam3+vlm_roi_filter",
        "vlm_used": true,
        "vlm_bbox": [
            249,
            139,
            277,
            355
        ],
        "vlm_bbox_used": [
            239,
            129,
            287,
            365
        ],
        "vlm_label": "white_tray_above_blue_dot",
        "kept_instance_ids": [
            1
        ],
        "source_instance_indices": [
            6
        ],
        "intersection_pixels": {
            "1": 3225
        },
        "sam3_raw_num_instances": 6
    },
    "detections": [
        {
            "instance_id": 1,
            "seg_score": 0.796875,
            "pose_score": 105.66796875,
            "pose_corrected": true,
            "reproj_error_px": 0.0,
            "reproj_error_before_px": 68.12540720995742,
            "bbox": [
                245,
                143,
                25,
                208
            ],
            "pose_4x4": [
                [
                    0.23474913835525513,
                    0.9696536064147949,
                    0.06829948723316193,
                    -0.053772769507566064
                ],
                [
                    0.6016563773155212,
                    -0.08975300937891006,
                    -0.793696403503418,
                    0.002931274253168315
                ],
                [
                    -0.7634804248809814,
                    0.22741246223449707,
                    -0.6044676303863525,
                    0.3230000138282776
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "t_m": [
                -0.053772769507566064,
                0.002931274253168315,
                0.3230000138282776
            ],
            "t_mm": [
                -53.77276950756606,
                2.931274253168315,
                323.0000138282776
            ],
            "R": [
                [
                    0.23474913835525513,
                    0.9696536064147949,
                    0.06829948723316193
                ],
                [
                    0.6016563773155212,
                    -0.08975300937891006,
                    -0.793696403503418
                ],
                [
                    -0.7634804248809814,
                    0.22741246223449707,
                    -0.6044676303863525
                ]
            ],
            "rotation_euler_zyx_rad": [
                2.78175333027043,
                0.8686851987977124,
                1.1987914503922514
            ],
            "xyzrxryrz": [
                -53.77276950756606,
                2.931274253168315,
                323.0000138282776,
                2.78175333027043,
                0.8686851987977124,
                1.1987914503922514
            ],
            "pose_path": "/home/ubuntu/stephen/01-code/FoundationPose/service_outputs/20260525_143152_65e77a3e/results/pose_inst01.txt"
        }
    ],
    "timing": {
        "seg_s": 7.120478913886473,
        "sam3_s": 7.120478913886473,
        "vlm_s": 1.126387930009514,
        "instance_filter_s": 0.029934188118204474,
        "pose_s": 0.7681358670815825,
        "vis_s": 0.02805963298305869,
        "pipeline_s": 9.072996532078832,
        "upload_s": 0.00042235804721713066,
        "total_s": 9.07341889012605
    },
    "sam3_prompt": "Plastic Reel"
}

```