#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/luka/transcriptor_vad_wav_chunking}"
SERVICE_NAME="${SERVICE_NAME:-transcriptor-vad.service}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_DIR"

echo "[deploy] Pull latest code"
git fetch --all --prune
git checkout main
git pull --ff-only origin main

echo "[deploy] Ensure venv"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[deploy] Restart service: $SERVICE_NAME"
sudo systemctl daemon-reload || true
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager -l

echo "[deploy] Done"
