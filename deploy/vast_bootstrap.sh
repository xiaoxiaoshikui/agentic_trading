#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/xiaoxiaoshikui/agentic_trading.git}"
BRANCH="${BRANCH:-main}"
WORKDIR="${WORKDIR:-/workspace/agentic_trading}"
RUN_MODE="${RUN_MODE:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[vast-bootstrap] repo=${REPO_URL} branch=${BRANCH} workdir=${WORKDIR} mode=${RUN_MODE}"

repair_vast_ssh_permissions() {
  mkdir -p /root/.ssh 2>/dev/null || true
  chown root:root /root /root/.ssh 2>/dev/null || true
  chmod go-w /root 2>/dev/null || true
  chmod 700 /root/.ssh 2>/dev/null || true
  if [ -f /root/.ssh/authorized_keys ]; then
    chown root:root /root/.ssh/authorized_keys 2>/dev/null || true
    chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
  fi
}

repair_vast_ssh_permissions
(
  for _ in $(seq 1 180); do
    repair_vast_ssh_permissions
    sleep 1
  done
) >/tmp/vast_bootstrap_ssh_repair.log 2>&1 &

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi

if [ ! -d "${WORKDIR}/.git" ]; then
  mkdir -p "$(dirname "${WORKDIR}")"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
else
  git -C "${WORKDIR}" fetch origin "${BRANCH}"
  git -C "${WORKDIR}" checkout "${BRANCH}"
  git -C "${WORKDIR}" pull --ff-only
fi

cd "${WORKDIR}"

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-research.txt

python -m unittest \
  test_tom_horizon_features.py \
  test_tom_calibration.py \
  test_tom_multi_agent_signal.py

case "${RUN_MODE}" in
  smoke)
    echo "[vast-bootstrap] smoke setup complete"
    ;;
  main)
    if [ ! -d data/multimodal_history_real ]; then
      echo "[vast-bootstrap] missing data/multimodal_history_real; copy data before RUN_MODE=main" >&2
      exit 2
    fi
    python -m experiments.run_mm_deep_experiment \
      --config experiments/configs/icaif_real_news_costaware_main.json \
      --device auto
    ;;
  shell)
    echo "[vast-bootstrap] environment ready; leaving instance idle for SSH"
    ;;
  *)
    echo "[vast-bootstrap] unknown RUN_MODE=${RUN_MODE}; expected smoke, main, or shell" >&2
    exit 2
    ;;
esac
