
This is the official implementation for the [paper](https://arxiv.org/pdf/2502.15679), "BOSS: Benchmark for Observation Space Shift in Long-Horizon Task".

# Table of Contents

- [Installation](#installation)
- [BOSS Benchmark](#boss-benchmark)
  - [Skills Training](#skills-training)
  - [Challenges](#challenges)
    - [BOSS-CH1: Single Predicate Modification](#boss-ch1-single-predicate-modification)
    - [BOSS-CH2: Accumulated Predicates Modification](#boss-ch2-accumulated-predicates-modification)
    - [BOSS-CH3: Real Long-Horizon Task](#boss-ch3-real-long-horizon-task)
- [Data Augmentation](#data-augmentation)
  - [RAMG](#ramg)
  - [Datasets](#datasets)
- [Citation](#citation)

# Installation

BOSS is adapted from [Libero](https://lifelong-robot-learning.github.io/LIBERO/html/getting_started/installation.html) and [OpenVLA](https://github.com/openvla/openvla/tree/main?tab=readme-ov-file#installation). 
Please create the conda environment and install packages by following the two repos.
```shell
conda create --name boss python=3.10
# For more details, please following the installation instructions in Libero & OpenVLA.
```

Download `assets` folder ([link](https://drive.google.com/file/d/1Rh24XyUy7Y5aE1jhiW2sZmpNh90-4s02/view?usp=sharing)), and put it under the root folder `./`.

Download [Libero-100](https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip), and put the `LIBERO-90` into the `./datasets` folder. 
Run the following command to rename it and only keep single skills to form the `boss_44` dataset.
```shell
python scripts/form_boss_44_dataset.py
```


# BOSS Benchmark
The BOSS benchmark contains 2 parts of codebases:
- `Skills Training`: Train single skills by using the 4 baseline algorithms: BC-RENET-RNN, BC-RESNET-T, BC-VIT-T, OpenVLA.
- `Challenges`: 3 challenges, including BOSS-CH1, BOSS-CH2, BOSS-CH3

## Skills Training
For BC-*, execute the following commands.
```shell
# For BC-RESNET-RNN
python libero/lifelong/train_skills.py policy="bc_rnn_policy"
# For BC-RESNET-T
python libero/lifelong/train_skills.py policy="bc_transformer_policy"
# For BC-VIT-T
python libero/lifelong/train_skills.py policy="bc_vilt_policy"
```
For OpenVLA, execute the following commands.
```shell
cd openvla
zsh shells/run_openvla.sh
```
The checkpoint will be saved at `runs/libero44/1.0.0/openvla-7b+libero44+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug`.

## Challenges

### BOSS-CH1: Single Predicate Modification
For BC-*, execute the following commands.
```shell
# Test BC-RESNET-RNN when unaffected by OSS
python libero/lifelong/eval_skills_unaffected_by_oss.py \
--benchmark "boss_44" \
--model_path_folder "./experiments/boss_44/0.0.0/BCRNNPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-RESNET-RNN when affected by OSS
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch1" \
--model_path_folder "./experiments/boss_44/0.0.0/BCRNNPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-RESNET-T when unaffected by OSS
python libero/lifelong/eval_skills_unaffected_by_oss.py \
--benchmark "boss_44" \
--model_path_folder "./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-RESNET-T when affected by OSS
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch1" \
--model_path_folder "./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-VIT-T when unaffected by OSS
python libero/lifelong/eval_skills_unaffected_by_oss.py \
--benchmark "boss_44" \
--model_path_folder "./experiments/boss_44/0.0.0/BCViLTPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-VIT-T when affected by OSS
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch1" \
--model_path_folder "./experiments/boss_44/0.0.0/BCViLTPolicy_seed10000/run_001/" \
--seed 10000
```
Checkpoints will be saved at `./experiments/boss_44/0.0.0/BC{algo}Policy_seed10000/run_00*/`.

For OpenVLA, execute the following commands.
```shell
# Test OpenVLA when unaffected by OSS
python experiments/robot/libero/eval_openvla_ch1_ch2.py  --seed 10000 --task_suite_name "boss_44"

# Test OpenVLA when affected by OSS
python experiments/robot/libero/eval_openvla_ch1_ch2.py  --seed 10000 --task_suite_name "ch1"
```

Checkpoints will be saved at `./experiments/logs/`.

### BOSS-CH2: Accumulated Predicates Modification
For BC-*, execute the following commands.
```shell
# Test BC-RESNET-RNN when affected by OSS - 2 modifications
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch2_2_modifications" \
--model_path_folder "./experiments/boss_44/0.0.0/BCRNNPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-RESNET-RNN when affected by OSS - 3 modifications
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch2_2_modifications" \
--model_path_folder "./experiments/boss_44/0.0.0/BCRNNPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-RESNET-T when affected by OSS - 2 modifications
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch2_2_modifications" \
--model_path_folder "./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-RESNET-T when affected by OSS - 3 modifications
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch2_3_modifications" \
--model_path_folder "./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-VIT-T when affected by OSS - 2 modifications
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch2_2_modifications" \
--model_path_folder "./experiments/boss_44/0.0.0/BCViLTPolicy_seed10000/run_001/" \
--seed 10000

# Test BC-VIT-T when affected by OSS - 3 modifications
python libero/lifelong/eval_skills_affected_by_oss.py \
--benchmark "ch2_3_modifications" \
--model_path_folder "./experiments/boss_44/0.0.0/BCViLTPolicy_seed10000/run_001/" \
--seed 10000
```

For OpenVLA, execute the following commands.
```shell
# Test OpenVLA when affected by OSS - 2 modifications
python experiments/robot/libero/eval_openvla_ch1_ch2.py  --seed 10000 --task_suite_name "ch2_2_modifications"

# Test OpenVLA when affected by OSS - 3 modifications
python experiments/robot/libero/eval_openvla_ch1_ch2.py  --seed 10000 --task_suite_name "ch2_3_modifications"
```

Checkpoints will be saved at `./experiments/logs/`.


### BOSS-CH3: Real Long-Horizon Task
For BC-*, execute the following commands.
```shell
python libero/lifelong/eval_skill_chain.py \
--seed 10000 --device_id 0 \
--model_path_folder "./experiments/boss_44/0.0.0/BCRNNPolicy_seed10000/run_001/"

python libero/lifelong/eval_skill_chain.py \
--seed 10000 --device_id 0 \
--model_path_folder "./experiments/boss_44/0.0.0/BCTransformerPolicy_seed10000/run_001/"

python libero/lifelong/eval_skill_chain.py \
--seed 10000 --device_id 0 \
--model_path_folder "./experiments/boss_44/0.0.0/BCViLTPolicy_seed10000/run_001/"
```

For OpenVLA, execute the following commands.
```shell
cd openvla
python experiments/robot/libero/eval_openvla_ch3.py --seed 10000
```
Checkpoints will be saved at `./experiments/logs/ch3/`.


# Data Augmentation


## RAMG


```shell
# Scale up the number of modified bddl files 
python RAMG/DA_bddl_files_scale_up_single_modification.py

# If you hope to include multiple modifications in a single bddl file, run the following command
python RAMG/DA_bddl_files_scale_up_multiple_modifications.py
```


## Datasets
After scaling up the number of modified bddl files, we can generate the augmented dataset based on boss_44 dataset and the large set of bddl files.
```shell
# Given boss_44 dataset and set of modified bddl files, generate the augmented dataset
python scripts/DA_demos_generation.py 
```
For OpenVLA, need to use `openvla/experiments/robot/libero/regenerate_libero_dataset.py` to regenerate no-op dataset, and [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) to convert to RLDS dataset. For more details, please check [README](https://github.com/openvla/openvla/tree/main?tab=readme-ov-file#openvla-an-open-source-vision-language-action-model) at OpenVLA Repo. 
To access our generated dataset, please contact `yygx@cs.unc.edu`. It contains 1727 tasks, each with 1 modification.
 

# Citation
If you use BOSS or find it interesting, please cite the following paper :)
```bibtex
@article{yang2025boss,
  title={BOSS: Benchmark for Observation Space Shift in Long-Horizon Task},
  author={Yang, Yue and Zhao, Linfeng and Ding, Mingyu and Bertasius, Gedas and Szafir, Daniel},
  journal={arXiv preprint arXiv:2502.15679},
  year={2025}
}
```