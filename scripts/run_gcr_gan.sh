#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/gcr_gan/dblp_v12.yaml}"
DEVICE_VALUE="${2:-}"
DEVICE_ARGS=()
if [[ -n "$DEVICE_VALUE" ]]; then
  DEVICE_ARGS=(--device "$DEVICE_VALUE")
fi

gcr-gan validate-data --config "$CONFIG_PATH"
gcr-gan prepare --config "$CONFIG_PATH" "${DEVICE_ARGS[@]}"
gcr-gan train --config "$CONFIG_PATH" "${DEVICE_ARGS[@]}"
gcr-gan evaluate --config "$CONFIG_PATH" "${DEVICE_ARGS[@]}"
