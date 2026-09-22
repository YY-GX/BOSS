# BOSS: Benchmark for Observation Space Shift in Long-Horizon Task

Official implementation of **BOSS** (Yang et al., *IEEE Robotics and Automation
Letters*, 2025) — [paper](https://arxiv.org/abs/2502.15679) ·
[project page](https://boss-benchmark.github.io/)

When a robot chains pre-trained skills, each skill changes the scene for the next
one. Often the change is irrelevant to whether the next skill *can* run — its
preconditions and effects are untouched — and yet it breaks that skill's
visuomotor policy anyway, because the policy has never seen the scene look like
that. We call this **Observation Space Shift (OSS)**, and BOSS measures it.

The benchmark is built on [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/)
and runs over 44 single-skill manipulation tasks.

| Challenge | Question it asks | Task suites |
|---|---|---|
| **BOSS-C1** | How much does *one* irrelevant change hurt? | `boss_44` vs `ch1` |
| **BOSS-C2** | Does the damage accumulate over several changes? | `ch2_2_modifications`, `ch2_3_modifications` |
| **BOSS-C3** | What does that cost a real 3-skill chain? | `ch3_1` … `ch3_10` |

Baselines: BC-RESNET-RNN, BC-RESNET-T, BC-VIT-T (in `libero/lifelong/`) and
OpenVLA (via [`integrations/openvla/`](integrations/openvla/README.md), which
wires BOSS into your own OpenVLA checkout).

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Data setup](#data-setup)
- [Pre-trained checkpoints](#pre-trained-checkpoints)
- [Running the challenges](#running-the-challenges)
  - [BOSS-C1: Single Predicate Shift](#boss-c1-single-predicate-shift)
  - [BOSS-C2: Accumulated Predicate Shift](#boss-c2-accumulated-predicate-shift)
  - [BOSS-C3: Skill Chaining](#boss-c3-skill-chaining)
  - [OpenVLA](#openvla)
- [Training from scratch](#training-from-scratch)
- [Data augmentation (RAMG)](#data-augmentation-ramg)
- [Reproducibility notes](#reproducibility-notes)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Citation](#citation)

## Quick start

Shortest path from a fresh clone to a BOSS number, using our released
checkpoints — no training required.

```shell
git clone https://github.com/Boss-Benchmark/BOSS.git && cd BOSS
conda env create -f environment.yml && conda activate boss
pip install -e .

# simulation assets and the 44 skill policies (~5 GB)
huggingface-cli download yygx/BOSS-assets --repo-type dataset --local-dir libero/libero/assets
huggingface-cli download yygx/BOSS-checkpoints --local-dir experiments/boss_44/0.0.0

# one BOSS-C3 chain
MUJOCO_GL=egl python libero/lifelong/eval_skill_chain.py \
  --model_path_folder ./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/ \
  --seed 10000 --max_steps 400 --lht 1
```

You do **not** need the demonstration datasets for this — they are only required
for training. Note `--max_steps 400`; see
[Reproducibility notes](#reproducibility-notes) for why it matters.

## Installation

Requires Linux, one CUDA GPU, and roughly 20 CPU cores — evaluation runs its 20
episodes as parallel subprocesses.

```shell
conda env create -f environment.yml    # creates the `boss` env (python 3.10)
conda activate boss
pip install -e .
```

`environment.yml` installs `requirements.txt`, which pins the versions the paper's
environment ran. OpenVLA is deliberately **not** part of it — its dependencies
conflict with these — so set that up separately if you need that baseline.

Then fetch the simulation assets (meshes, textures, scene XMLs) from
[`yygx/BOSS-assets`](https://huggingface.co/datasets/yygx/BOSS-assets):

```shell
huggingface-cli download yygx/BOSS-assets --repo-type dataset \
  --local-dir libero/libero/assets
```

On first import the package writes `.boss/config.yaml` at the repository root,
recording where assets, bddl files, init states and datasets live. It is
machine-local and git-ignored. Run all commands below **from the repository root**.

## Data setup

Only needed for training; skip it if you are evaluating released checkpoints.

```shell
# LIBERO-90 demonstrations (~20 GB)
python benchmark_scripts/download_libero_datasets.py --datasets libero_90

# reduce to the 44 single-skill BOSS tasks
python scripts/form_boss_44_dataset.py
```

The first command pulls from upstream's HuggingFace mirror
(`yifengzhu-hf/LIBERO-datasets`); LIBERO's original utexas.box.com links now
return 403.

The second renames the folder to `libero/datasets/boss_44` and moves every demo
that is not one of the 44 BOSS tasks to `libero/datasets/libero_90_unused/`
(`--delete` removes them instead, `--yes` skips the prompt). The keep-list is
derived from `boss_task_map["boss_44"]`, so it cannot drift from the benchmark
definition.

## Pre-trained checkpoints

Skill policies for all three BC baselines — 44 checkpoints each, seed 10000 — are
published at [`yygx/BOSS-checkpoints`](https://huggingface.co/yygx/BOSS-checkpoints):

```shell
huggingface-cli download yygx/BOSS-checkpoints --local-dir experiments/boss_44/0.0.0
```

That lands each policy at `experiments/boss_44/0.0.0/<PolicyType>_seed10000/run_001/`,
where `<PolicyType>` is `BCRNNPolicy`, `BCTransformerPolicy` or `BCViLTPolicy`.

The two fine-tuned OpenVLA models from Table I are at
[`yygx/BOSS-openvla-adapters`](https://huggingface.co/yygx/BOSS-openvla-adapters).

## Running the challenges

Pick one checkpoint folder:

```shell
export MODEL=./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/
```

Every evaluation rolls out 20 episodes per task and reports a success rate. Add
`--max_steps 400` to match the paper, and `--device_id N` to pick a GPU.

### BOSS-C1: Single Predicate Shift

Each of the 44 skills is evaluated twice: on its original task, and on a
counterpart where a preceding skill has changed one irrelevant predicate.

```shell
# unaffected by OSS
python libero/lifelong/eval_skills_unaffected_by_oss.py \
  --benchmark boss_44 --model_path_folder $MODEL --seed 10000 --max_steps 400

# affected by OSS
python libero/lifelong/eval_skills_affected_by_oss.py \
  --benchmark ch1 --model_path_folder $MODEL --seed 10000 --max_steps 400
```

| | |
|---|---|
| **Output** | `$MODEL/eval_tasks_on_ori_envs_seed10000/` and `$MODEL/eval_tasks_on_modified_envs_seed10000/` — per-task `.stats`, `succ_list_*.npy`, a `succ_per_task*.json` keyed by task id, and rollout videos |
| **Metric** | Ratio Performance Delta, `RPD = (unaffected − affected) / unaffected`, per task — paper Fig. 3 |
| **Runtime** | a few minutes per task on one GPU, so a few hours for all 44 |

### BOSS-C2: Accumulated Predicate Shift

The same 44 skills, against counterparts carrying two and three accumulated
modifications.

```shell
python libero/lifelong/eval_skills_affected_by_oss.py \
  --benchmark ch2_2_modifications --model_path_folder $MODEL --seed 10000 --max_steps 400

python libero/lifelong/eval_skills_affected_by_oss.py \
  --benchmark ch2_3_modifications --model_path_folder $MODEL --seed 10000 --max_steps 400
```

RPD is computed against the same C1-unaffected run. Paper Fig. 4 plots the average
positive RPD and the fraction of tasks where OSS occurs, for one, two and three
modifications.

### BOSS-C3: Skill Chaining

Ten hand-built chains of three skills each, executed end to end.

```shell
# all ten chains
python libero/lifelong/eval_skill_chain.py \
  --model_path_folder $MODEL --seed 10000 --max_steps 400

# or a subset
python libero/lifelong/eval_skill_chain.py \
  --model_path_folder $MODEL --seed 10000 --max_steps 400 --lht 1 2 3
```

| | |
|---|---|
| **Output** | `$MODEL/long_horizon_task_ch3_<i>_seed10000/` — `*.stats` (overall success rate), `level_succ.pkl` (fraction reaching each skill in the chain) and videos |
| **Metric** | Delta to Upper Bound Ratio, `DUBR = (upper_bound − actual) / upper_bound`, where `upper_bound` is the product of that chain's per-skill success rates from the C1-unaffected run — paper Fig. 5 |
| **Runtime** | roughly 40 minutes per chain on one GPU at `--max_steps 400` |

### OpenVLA

Set it up once ([`integrations/openvla/`](integrations/openvla/README.md)), then
from inside your OpenVLA checkout:

```shell
SEED=10000 bash ../integrations/openvla/shells/eval_openvla.sh
```

That covers C1, C2 and C3; logs and rollouts go to `experiments/logs/`.

## Training from scratch

```shell
python libero/lifelong/train_skills.py policy=bc_rnn_policy         seed=10000
python libero/lifelong/train_skills.py policy=bc_transformer_policy seed=10000
python libero/lifelong/train_skills.py policy=bc_vilt_policy        seed=10000
```

Checkpoints land in `experiments/boss_44/0.0.0/<PolicyType>_seed<seed>/run_<NNN>/`,
one `task<i>_model.pth` per skill. The paper averages over three training runs per
baseline (`seed=10000`, `10001`, `10002`).

Useful overrides: `is_debug=true` shortens a run to 2 tasks / 2 epochs for a smoke
test, `use_wandb=true` enables Weights & Biases, `device=cuda:N` picks a GPU.

For OpenVLA, after [setting it up](integrations/openvla/README.md), from inside
that checkout:

```shell
DATASET_NAME=libero44 bash ../integrations/openvla/shells/finetune_openvla.sh
```

## Data augmentation (RAMG)

The Rule-based Automatic Modification Generator scales the 44 tasks into 1,727
visually modified variants. **The generated task definitions already ship here**
(`libero/libero/bddl_files/data_augmentation/`, 1,727 files), so step 1 is only
needed to regenerate them or change the modification budget.

```shell
# 1. (optional) regenerate the modified bddl files
python RAMG/DA_bddl_files_scale_up_single_modification.py
#    ... or with several modifications per file
python RAMG/DA_bddl_files_scale_up_multiple_modifications.py

# 2. replay the boss_44 demos in the modified environments
python RAMG/DA_demos_generation.py --benchmark data_augmentation
```

Step 2 replays every original demonstration in its modified environment and keeps
the ones that still succeed, so the trajectory is unchanged and only the visual
observation differs. It takes roughly 6 minutes per task and produces a few
hundred MB each; `--start-index N` resumes an interrupted run.

Train on the augmented set (paper Table I, **Setup B**):

```shell
python libero/lifelong/train_skills_augmented.py policy=bc_transformer_policy seed=10000
```

This combines each task's original demonstration with its augmented variants; see
the `augmentation_*` keys in `libero/configs/config.yaml`.

### Pre-generated data

The RLDS/TFDS build used for the OpenVLA results — 56,945 episodes over 1,727
modified tasks — is published at
[`yygx/BOSS-data-augmentation`](https://huggingface.co/datasets/yygx/BOSS-data-augmentation):

```shell
huggingface-cli download yygx/BOSS-data-augmentation --repo-type dataset \
  --local-dir datasets
```

The HDF5 form the BC baselines consume is **not** distributed; regenerate it with
step 2 above, which needs only `boss_44` and the bddl files already in this
repository. For OpenVLA you additionally need
`integrations/openvla/regenerate_libero_dataset.py` and
[rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) to build
the RLDS form yourself.

## Reproducibility notes

**Use `--max_steps 400`.** Evaluation hyper-parameters are read from the
checkpoint, not from `libero/configs/eval/default.yaml`, and the released
checkpoints carry `max_steps=600` while the published results used 400. On
BOSS-C3 chain 1, BC-RESNET-T scores a Delta to Upper Bound Ratio of **89 % at 400
steps** — the value in Figure 5 — and only 67 % at 600, because the extra time
lets more chains finish.

**What the seeds control.** Three different things are seeded, and they are easy
to confuse:

| | Controlled by | Effect |
|---|---|---|
| Policy initialisation & data order | `seed=` in training | The main source of run-to-run variance. The paper averages three training runs per baseline. |
| Action sampling at evaluation | `--seed` in the eval scripts | The BC policies use a GMM head and *sample* each action, so evaluation is stochastic. The scripts call `control_seed(args.seed)`, which makes a given command reproducible. |
| Environment randomisation | `cfg.seed` via `env.seed()` | Effectively inert: every episode starts from a stored `.pruned_init` state that is written over the whole MuJoCo state, and C3 sub-task transitions carry the previous skill's state forward the same way. |

**Environment drift.** From PyTorch 2.6 `torch.load` defaults to
`weights_only=True`, which cannot unpickle the `EasyDict` config stored inside our
checkpoints; `torch_load_model` passes `weights_only=False` so older checkpoints
keep loading.

**Known limitations.**
- `libero/libero/benchmark/__init__.py` registers suites beyond those used in the
  paper (`g1`–`g8`, `libero_local*`, `factor_1`, `factor_2`). They are exploratory
  and not part of the published results.
- The augmented demonstrations for Table I, Setup B are distributed in RLDS form
  only; the HDF5 form has to be regenerated with RAMG.

## Troubleshooting

**`MUJOCO_GL` / EGL errors, or rendering hangs on a headless machine.** Export
`MUJOCO_GL=egl` before running anything that renders.

**CUDA out of memory.** Evaluation loads one policy copy per parallel episode.
Pass `--device_id N` to move to a free GPU; check `nvidia-smi` first.

**Paths point at someone else's machine.** Delete `.boss/config.yaml` and re-run;
it is regenerated from the current checkout. `BOSS_CONFIG_PATH` relocates it.

**`wandb.errors.Error: You must call wandb.init()`.** Training only logs to
Weights & Biases when `use_wandb=true`; if you enabled it, run `wandb login` first.

**Evaluation reports a suspiciously small number of tasks.** `--is_debug 1`
evaluates a single task. It defaults to 0; pass `--is_debug 0` explicitly if a
shell alias or script sets it.

**`ModuleNotFoundError: libero`.** Run from the repository root, and check that
`pip install -e .` succeeded.

**Install fails building `bddl` with `[Errno 39] Directory not empty`.** `bddl`
ships as a source distribution, so pip builds it locally, and the cleanup step of
that build fails on NFS — a deleted-but-open file is renamed to `.nfsXXXX`, which
leaves the directory non-empty. Build on a local disk instead:

```shell
TMPDIR=/var/tmp conda env create -f environment.yml
```

The environment itself can live on NFS; only the build directory needs to be
local.

## Repository layout

```
libero/
  configs/        hydra configs (policy, training, evaluation)
  mappings/       task-language mappings per suite
  libero/         environments, task suites, bddl files, init states
  lifelong/       training and evaluation entry points, policies
RAMG/             Rule-based Automatic Modification Generator
scripts/          dataset preparation and demonstration replay
integrations/
  openvla/        BOSS-side files plus an installer for your OpenVLA checkout
openvla/          empty; clone OpenVLA here
tests/            regression tests
```

## Tests

```shell
pytest tests/
```

A regression suite covering the seeding, path-resolution, packaging and
dataset-preparation behaviour. It runs in well under a minute and needs neither
assets nor checkpoints.

## Citation

```bibtex
@article{yang2025boss,
  title={BOSS: Benchmark for observation space shift in long-horizon task},
  author={Yang, Yue and Zhao, Linfeng and Ding, Mingyu and Bertasius, Gedas and Szafir, Daniel},
  journal={IEEE Robotics and Automation Letters},
  volume={10},
  number={9},
  pages={8882--8889},
  year={2025},
  publisher={IEEE}
}
```

Built on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (Liu et al.,
2023) and [OpenVLA](https://github.com/openvla/openvla) (Kim et al., 2024).
