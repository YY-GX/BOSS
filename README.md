# Installation
TODO
```shell
conda create -n libero python=3.10.0
# To ensure pip installation location is same as conda
conda install --force-reinstall pip

# Libero related packages
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install robosuite
cd ./YourLocation/Libero && pip install -e .
```

```shell
pip install --force-reinstall 
```

TODO: move `datasets` folder and `assets` folder.

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

## Challenges

### BOSS-CH1: Single Predicate Modification
For BC-*, execute the following commands.
```shell
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
```
Checkpoints will be saved at `./experiments/boss_44/0.0.0/BC{algo}Policy_seed10000/run_00*/`.

For OpenVLA, execute the following commands.
```shell
python experiments/robot/libero/run_libero44_eval_args.py  --seed 10000 --task_suite_name "boss_44"
```



### BOSS-CH2: Accumulated Predicates Modification
```shell

```


### BOSS-CH3: Real Long-Horizon Task
```shell
python libero/lifelong/eval_skill_chain.py \
--seed 10000 --device_id 0 \
--model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/skill_policies_without_wrist_camera_view/Sequential/BCRNNPolicy_seed10000/all"
```

