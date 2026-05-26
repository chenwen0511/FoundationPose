#!/usr/bin/env bash
# 白盘测试：离线推理（需 foundationpose 环境 + GPU + 在线 SAM3 服务）
set -euo pipefail
cd "$(dirname "$0")/.."
export GENPOSE2_SAM3_API_URL="${GENPOSE2_SAM3_API_URL:-http://127.0.0.1:18002/infer}"
exec python test/run_local_infer.py "$@"
