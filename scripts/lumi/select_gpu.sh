#!/usr/bin/env bash
set -euo pipefail

# LUMI-G exposes one MI250X GCD per local rank.
export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"
export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
exec "$@"
