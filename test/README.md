# 测试样例 — Plastic Reel

## 数据文件

```
test/
├── CAD/
│   └── tray_180mm_centered_mesh_v2.ply   # 托盘 CAD，顶点单位 mm
└── 20260509_145205_5a80ad10/inputs/      # 单实例样例
    ├── rgb.png
    ├── depth.png
    └── camera.json
```

## 关键配置

| 项 | 值 |
|----|-----|
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
python http_server.py --port 8002
```

## 调用测试

```bash
python test/call_infer.py --sample-dir test/20260509_145205_5a80ad10/inputs
python test/run_local_infer.py --sample-dir test/20260509_145205_5a80ad10/inputs
bash test/run_test.sh
```

## 输出

`service_outputs/<request_id>/results/`：

- `vis_ism.png` — SAM3 分割（Plastic Reel）
- `vis_pose.png` — 3D 框 + 坐标轴
- `detection_pose.json` — 位姿列表
