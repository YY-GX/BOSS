#!/usr/bin/env bash
# Evaluate a fine-tuned OpenVLA on all three BOSS challenges.
# Run from inside your OpenVLA checkout, after `python <boss>/integrations/openvla/install.py`.
set -euo pipefail

SEED="${SEED:-10000}"

for suite in boss_44 ch1 ch2_2_modifications ch2_3_modifications; do
  echo ">> BOSS-C1/C2 :: ${suite}"
  python experiments/robot/libero/eval_openvla_ch1_ch2.py --seed "${SEED}" --task_suite_name "${suite}"
done

echo ">> BOSS-C3 :: skill chaining"
python experiments/robot/libero/eval_openvla_ch3.py --seed "${SEED}"
