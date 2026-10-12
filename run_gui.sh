#!/usr/bin/env bash
# Launch the detection GUI. Usage: bash run_gui.sh [port]
# Set GUI_PYTHON to choose an interpreter; otherwise python3 from PATH is used.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-7860}"
PYTHON_BIN="${GUI_PYTHON:-python3}"
if ! "$PYTHON_BIN" -c 'import fastapi, uvicorn, PIL, cv2, shapely' 2>/dev/null; then
  echo "Missing GUI dependencies for $PYTHON_BIN" >&2
  echo "Install them with: $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 1
fi
exec "$PYTHON_BIN" -c "
from app import config
config.PORT = int('$PORT')
from app.main import main
main()
"
