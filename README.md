# Installation

```shell
cd BOSS
conda env create -f environment.yaml
```

TODO: move `datasets` folder and `assets` folder.

For openvla installation, please check 
```shell

pip install --upgrade transformers
```


# BOSS Benchmark
The BOSS benchmark contains 2 parts of codebases:
- Skills Training: Train single skills by using the 4 baseline algorithms: BC-RENET-RNN, BC-RESNET-T, BC-VIT-T, OpenVLA.
- Challenges: 3 challenges, including BOSS-CH1, BOSS-CH2, BOSS-CH3

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
cd openvla/shells
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
python experiments/robot/libero/eval_openvla_ch3.py --seed 10000
```
Checkpoints will be saved at `./experiments/logs/ch3/`.


# Data Augmentation

## RAMG

```shell
TODO
```

## Datasets
TODO
