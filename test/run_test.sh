#!/usr/bin/env bash
# 白盘测试：离线推理（需 foundationpose 环境 + GPU + SAM3）
set -euo pipefail
cd "$(dirname "$0")/.."
export GENPOSE2_SAM3_ROOT="${GENPOSE2_SAM3_ROOT:-/home/ubuntu/stephen/01-code/sam3}"
exec python test/run_local_infer.py "$@"
