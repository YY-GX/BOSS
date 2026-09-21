# OpenVLA integration

BOSS does not vendor OpenVLA. Clone it yourself and run one installer; this folder
holds only the BOSS-specific pieces.

## Setup

```shell
git clone https://github.com/openvla/openvla.git openvla
cd openvla && pip install -e . && cd ..

python integrations/openvla/install.py            # --openvla-dir to point elsewhere
```

The installer is idempotent and `--check` reports what it would do without writing.
It:

1. copies the five files below into `openvla/experiments/robot/libero/`;
2. registers the `libero44` and `libero_bl3_all` RLDS datasets in
   `prismatic/vla/datasets/rlds/oxe/{configs,transforms}.py`, so
   `--dataset_name libero44` resolves during fine-tuning.

| File | Role |
|---|---|
| `eval_openvla_ch1_ch2.py` | BOSS-C1 and BOSS-C2 evaluation |
| `eval_openvla_ch3.py` | BOSS-C3 skill-chaining evaluation |
| `eval_openvla_speed_up.py` | C1/C2 evaluation parallelised over a vector env |
| `libero_utils.py` | OpenVLA's helper plus `get_libero_subproc_env` |
| `regenerate_libero_dataset.py` | Rewrites BOSS demos into OpenVLA's no-op-filtered format |

## Fine-tuning

Convert the demonstrations to RLDS with
[rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) after
running `regenerate_libero_dataset.py`, then, from inside the OpenVLA checkout:

```shell
# paper Table I, Setup A
DATASET_NAME=libero44 bash ../integrations/openvla/shells/finetune_openvla.sh
# paper Table I, Setup B (RAMG-augmented)
DATASET_NAME=libero_bl3_all bash ../integrations/openvla/shells/finetune_openvla.sh
```

Set `WANDB_ENTITY` to enable Weights & Biases logging; it is off by default.

## Evaluation

```shell
SEED=10000 bash ../integrations/openvla/shells/eval_openvla.sh
```

Logs and rollouts are written to `experiments/logs/`.

## Seeding

All three evaluation scripts now seed from `--seed`:
`eval_openvla_ch1_ch2.py` already called `set_seed_everywhere`;
`eval_openvla_ch3.py` imported it but never called it, and hardcoded `seed=10000`
in the model config; `get_libero_subproc_env` hardcoded `env.seed(0)`.

`eval_openvla_ch3.py` also takes `--lht` to evaluate a subset of the ten chains
instead of always looping over all of them.

See the reproducibility notes in the top-level README for what each seed actually
controls — in particular, the environment seed has no effect on rollouts, because
every episode starts from a stored `.pruned_init` state.
