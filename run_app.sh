#!/usr/bin/env bash
# Запуск русскоязычного интерфейса «до/после» (TPS, CPU).
set -euo pipefail
cd "$(dirname "$0")"

export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Не найдено виртуальное окружение .venv." >&2
  echo "Создайте его: python -m venv .venv && .venv/Scripts/python.exe -m pip install -e \".[app]\"" >&2
  exit 1
fi

exec "$PY" app_before_after.py "$@"
