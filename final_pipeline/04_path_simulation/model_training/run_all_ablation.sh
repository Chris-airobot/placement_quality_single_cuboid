#!/usr/bin/env bash
set -euo pipefail

# Runs five cases:
# 1) all_on        : USE_META=1, USE_DELTA=1, USE_CORNERS=1, USE_TRANSPORT=1
# 2) no_meta       : USE_META=0, others on
# 3) no_trans      : USE_DELTA=0, others on
# 4) no_corner     : USE_CORNERS=0, others on
# 5) no_frame      : USE_TRANSPORT=0, others on

PYTHON=${PYTHON:-python3}
export PYTHONPATH="/home/chris/Chris/placement_ws/src:${PYTHONPATH:-}"
MOD_PATH="placement_quality.path_simulation.model_training.train"

run_case() {
  local name="$1"
  local use_meta="$2"
  local use_delta="$3"
  local use_corners="$4"
  local use_transport="$5"

  local out_dir="/home/chris/Chris/placement_ws/src/data/box_simulation/v7/training_out/${name}"
  mkdir -p "${out_dir}"
  echo "\n===== Running case: ${name} =====" | tee "${out_dir}/run.log"
  ${PYTHON} - <<PY 2>&1 | tee -a "${out_dir}/run.log"
import importlib
mod = importlib.import_module("${MOD_PATH}")

# Configure toggles for this run
mod.USE_META = bool(${use_meta})
mod.USE_DELTA = bool(${use_delta})
mod.USE_CORNERS = bool(${use_corners})
mod.USE_TRANSPORT = bool(${use_transport})

# Ensure we precompute and evaluate deck automatically
mod.USE_PRECOMPUTED = True
mod.AUTO_PRECOMPUTE = True
mod.AUTO_EVAL_DECK = True

# Separate output directory per case
mod.OUT_DIR = f"${out_dir}"

# Compose matching precompute root after toggles are set
mod.PRECOMP_ROOT = mod._compose_precomp_root()

print("Toggles:", dict(USE_META=mod.USE_META, USE_DELTA=mod.USE_DELTA, USE_CORNERS=mod.USE_CORNERS, USE_TRANSPORT=mod.USE_TRANSPORT))
mod.main()
PY
}

# 1) All toggles enabled
# run_case all_on 1 1 1 1

# 2) Meta off
run_case no_meta 0 0 1 0

# # 3) Relative translation off (use absolute)
run_case no_trans 1 0 1 1

# # 4) Corners off
run_case no_corner 1 0 0 0

# # 5) Transport frame off (use world frame)
run_case no_frame 1 1 1 0

# run_case all_on 1 0 1 0


echo "\nAll runs completed."


