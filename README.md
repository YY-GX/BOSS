# BOSS: Benchmark for Observation Space Shift in Long-Horizon Task

Official implementation of [BOSS](https://arxiv.org/pdf/2502.15679) (Yang et al., 2025).
Project page: https://boss-benchmark.github.io/

BOSS studies **Observation Space Shift (OSS)**: when skills are chained, a preceding
skill changes the scene in ways that break the next skill's visuomotor policy, even
though the change is irrelevant to that skill's preconditions and effects. The
benchmark is built on [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/) and
ships three challenges over 44 single-skill tasks.

| Challenge | What it measures | Task suites |
|---|---|---|
| **BOSS-C1** | Single predicate shift | `boss_44` (unaffected) vs `ch1` (modified) |
| **BOSS-C2** | Accumulated predicate shift | `ch2_2_modifications`, `ch2_3_modifications` |
| **BOSS-C3** | Real long-horizon skill chaining | `ch3_1` … `ch3_10` |

Baselines: BC-RESNET-RNN, BC-RESNET-T, BC-VIT-T (in `libero/lifelong/`) and OpenVLA
(via [`integrations/openvla/`](integrations/openvla/README.md), which wires BOSS
into your own OpenVLA checkout).

## Table of Contents

- [Installation](#installation)
- [Data setup](#data-setup)
- [Skills training](#skills-training)
- [Challenges](#challenges)
  - [BOSS-C1: Single Predicate Shift](#boss-c1-single-predicate-shift)
  - [BOSS-C2: Accumulated Predicate Shift](#boss-c2-accumulated-predicate-shift)
  - [BOSS-C3: Real Long-Horizon Task](#boss-c3-real-long-horizon-task)
- [Data augmentation (RAMG)](#data-augmentation-ramg)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)

## Installation

BOSS is adapted from [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html).
The three BC baselines need only the environment below; OpenVLA is optional and
set up separately (see [`integrations/openvla/`](integrations/openvla/README.md)).

```shell
conda env create -f environment.yml    # creates the `boss` env (python 3.10)
conda activate boss
pip install -e .
```

`environment.yml` pins exact linux-64 builds. On other platforms, create the env
manually and install `requirements.txt` instead.

Then download the [`assets` folder](https://drive.google.com/file/d/1Rh24XyUy7Y5aE1jhiW2sZmpNh90-4s02/view?usp=sharing)
and unpack it to `libero/libero/assets/`.

On first import the package writes `.boss/config.yaml` at the repository root,
recording where assets, bddl files, init states and datasets live. It is
machine-local and git-ignored; delete it to regenerate, or point `BOSS_CONFIG_PATH`
somewhere else. Run all commands below **from the repository root**.

## Data setup

Download the LIBERO-90 demonstrations into `libero/datasets/libero_90/`. LIBERO's
original utexas.box.com links now return 403, so use the HuggingFace mirror:

```shell
pip install huggingface_hub
huggingface-cli download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset --include "libero_90/*" \
  --local-dir libero/datasets
```

Then reduce it to the 44 single-skill tasks:

```shell
python scripts/form_boss_44_dataset.py
```

This renames the downloaded folder to `libero/datasets/boss_44` and moves every demo
that is not one of the 44 BOSS tasks to `libero/datasets/libero_90_unused/`
(pass `--delete` to remove them instead, `--yes` to skip the confirmation prompt).
The keep-list is derived from `boss_task_map["boss_44"]`, so it cannot drift from the
benchmark definition.

## Skills training

```shell
# BC-RESNET-RNN
python libero/lifelong/train_skills.py policy=bc_rnn_policy seed=10000
# BC-RESNET-T
python libero/lifelong/train_skills.py policy=bc_transformer_policy seed=10000
# BC-VIT-T
python libero/lifelong/train_skills.py policy=bc_vilt_policy seed=10000
```

Checkpoints land in `experiments/boss_44/0.0.0/<PolicyType>_seed<seed>/run_<NNN>/`,
one `task<i>_model.pth` per skill. `<PolicyType>` is `BCRNNPolicy`,
`BCTransformerPolicy` or `BCViLTPolicy`. The paper averages over three training runs
(`seed=10000`, `10001`, `10002`) — see [Reproducibility notes](#reproducibility-notes).

For OpenVLA, set it up once
([`integrations/openvla/`](integrations/openvla/README.md)) and then, from inside
your OpenVLA checkout:

```shell
DATASET_NAME=libero44 bash ../integrations/openvla/shells/finetune_openvla.sh
```

The adapter is written to `runs/libero44/1.0.0/openvla-7b+libero44+...`.

## Challenges

The examples below evaluate one checkpoint folder; substitute the policy and seed
you want.

```shell
export MODEL=./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/
```

### BOSS-C1: Single Predicate Shift

```shell
# unaffected by OSS (the 44 original skills)
python libero/lifelong/eval_skills_unaffected_by_oss.py \
  --benchmark boss_44 --model_path_folder $MODEL --seed 10000

# affected by OSS (the 44 single-modification counterparts)
python libero/lifelong/eval_skills_affected_by_oss.py \
  --benchmark ch1 --model_path_folder $MODEL --seed 10000
```

Results are written under `$MODEL/eval_tasks_on_ori_envs_seed<seed>/` and
`$MODEL/eval_tasks_on_modified_envs_seed<seed>/` respectively.
Ratio Performance Delta (paper Fig. 3) is computed from these two runs.

### BOSS-C2: Accumulated Predicate Shift

```shell
python libero/lifelong/eval_skills_affected_by_oss.py \
  --benchmark ch2_2_modifications --model_path_folder $MODEL --seed 10000

python libero/lifelong/eval_skills_affected_by_oss.py \
  --benchmark ch2_3_modifications --model_path_folder $MODEL --seed 10000
```

### BOSS-C3: Real Long-Horizon Task

```shell
# all ten chains
python libero/lifelong/eval_skill_chain.py \
  --model_path_folder $MODEL --seed 10000 --device_id 0

# or a subset
python libero/lifelong/eval_skill_chain.py \
  --model_path_folder $MODEL --seed 10000 --device_id 0 --lht 1 2 3
```

Per-chain success rate and per-level success rate land in
`$MODEL/long_horizon_task_ch3_<i>_seed<seed>/`. Delta to Upper Bound Ratio
(paper Fig. 5) is derived from these together with the C1-unaffected numbers.

For OpenVLA, from inside your OpenVLA checkout:

```shell
SEED=10000 bash ../integrations/openvla/shells/eval_openvla.sh
```

That covers C1, C2 and C3; logs and rollouts go to `experiments/logs/`.

## Data augmentation (RAMG)

The Rule-based Automatic Modification Generator scales the 44 tasks into a large set
of visually modified variants (1,727 tasks with one modification each).

```shell
# 1. generate modified bddl files
python RAMG/DA_bddl_files_scale_up_single_modification.py
# ... or with several modifications per file
python RAMG/DA_bddl_files_scale_up_multiple_modifications.py

# 2. replay the boss_44 demos in the modified environments
python RAMG/DA_demos_generation.py --benchmark data_augmentation
```

Step 2 is long-running; `--start-index N` resumes from the N-th boss_44 task.

Train on the augmented set (paper Table I, **Setup B**):

```shell
python libero/lifelong/train_skills_augmented.py policy=bc_transformer_policy seed=10000
```

It combines each task's original demo with its augmented variants; see the
`augmentation_*` keys in `libero/configs/config.yaml`. Note that
`augmentation_mapping` must name a json in `libero/mappings/`.

For OpenVLA, regenerate a no-op-filtered dataset with
`integrations/openvla/regenerate_libero_dataset.py` and convert it with
[rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder).

To request our pre-generated augmented dataset, contact `yygx@cs.unc.edu`.

## Reproducibility notes

**What the seeds control.** Three different things are seeded, and they are often
confused:

| | Controlled by | Effect |
|---|---|---|
| Policy initialisation & data order | `seed=` in training | The main source of run-to-run variance. The paper averages over `10000`, `10001`, `10002`, i.e. **three training runs per baseline**. |
| Action sampling at evaluation | `--seed` in the eval scripts | The BC policies use a GMM head and *sample* each action, so evaluation is stochastic. The eval scripts now call `control_seed(args.seed)`, which makes a given command reproducible. |
| Environment randomisation | `cfg.seed` via `env.seed()` | Effectively inert: every rollout starts from a stored `.pruned_init` state that is written over the whole MuJoCo state, and sub-task transitions in C3 carry the previous skill's state forward the same way. |

Evaluations before this was fixed were unseeded, so repeated runs of the same command
returned different numbers. Seeding does not bias the results — it fixes one draw from
the same distribution — but individual numbers will not match an unseeded run.

**Evaluation hyper-parameters come from the checkpoint**, not from
`libero/configs/eval/default.yaml`: the eval scripts read `cfg` out of
`task<i>_model.pth`, so `n_eval` and `max_steps` are whatever was configured at
training time. Editing the yaml afterwards has no effect on evaluation.

**Environment drift.** The paper numbers were produced with the pinned
`environment.yml`. Notably, from PyTorch 2.6 `torch.load` defaults to
`weights_only=True`, which cannot unpickle the `EasyDict` config stored inside our
checkpoints; `torch_load_model` passes `weights_only=False` to keep old checkpoints
loadable.

**Known limitations.**
- `libero/libero/benchmark/__init__.py` registers suites beyond those used in the
  paper (`g1`–`g8`, `libero_local*`, `factor_1`, `factor_2`). They are exploratory and
  not part of the published results.
- The OpenVLA evaluations seed the environment with a literal `0`, so unlike the
  BC scripts they do not vary with `--seed`.

## Tests

```shell
pytest tests/
```

A small regression suite covering the seeding, path-resolution and dataset-prep
fixes. It runs in about 30 seconds and needs neither assets nor checkpoints.

## Citation

```bibtex
@article{yang2025boss,
  title={BOSS: Benchmark for Observation Space Shift in Long-Horizon Task},
  author={Yang, Yue and Zhao, Linfeng and Ding, Mingyu and Bertasius, Gedas and Szafir, Daniel},
  journal={arXiv preprint arXiv:2502.15679},
  year={2025}
}
```
