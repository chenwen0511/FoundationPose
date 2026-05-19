# 测试样例 — 白盘 (white plate)

## 数据文件

```
test/
├── CAD/
│   └── tray_180mm_centered_mesh_v2.ply   # 托盘 CAD，顶点单位 mm
└── 20260507_105248_1d1db1bb/inputs/
    ├── rgb.png       # 640×480 RGB
    ├── depth.png     # 640×480 uint16，单位 mm
    └── camera.json   # 相机内参 + depth_scale
```

## 关键配置

| 项 | 值 |
|----|-----|
| CAD | `test/CAD/tray_180mm_centered_mesh_v2.ply` |
| CAD 缩放 | `FOUNDATIONPOSE_MESH_SCALE=0.001`（mm → m） |
| SAM3 提示词 | `"white plate"` |

## 启动服务

```bash
conda activate foundationpose

python http_server.py --host 0.0.0.0 --port 8002
# 默认已使用 test/CAD/tray_180mm_centered_mesh_v2.ply、mesh_scale=0.001、prompt="white plate"
```

或显式指定：

```bash
export FOUNDATIONPOSE_MESH_FILE=test/CAD/tray_180mm_centered_mesh_v2.ply
export FOUNDATIONPOSE_MESH_SCALE=0.001
export GENPOSE2_SAM3_PROMPT="white plate"
export GENPOSE2_SAM3_ROOT=/home/ubuntu/stephen/01-code/sam3
python http_server.py --port 8002
```

## 调用测试

```bash
# HTTP 客户端（需先启动 http_server.py）
python test/call_infer.py

# 离线本地推理（GPU + SAM3 + 权重）
python test/run_local_infer.py

# 一键脚本
bash test/run_test.sh
```

## camera.json

- `depth.png` 为 uint16 毫米；`depth_scale=1.0` 时服务会对 PNG 深度自动 ×0.001 转为米。

## 输出

`service_outputs/<request_id>/results/`：

- `vis_ism.png` — SAM3 分割（white plate）
- `vis_pose.png` — 各实例 3D 框 + 坐标轴
- `detection_pose.json` — 位姿列表
