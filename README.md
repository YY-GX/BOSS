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

## Train skills
For BC-*, execute the following commands.
```shell
# For BC-RESNET-T
python libero/lifelong/train_skills.py policy="bc_transformer_policy"
```
For OpenVLA, execute the following commands.
```shell

```

## Challenge 1
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


## Challenge 2



## Challenge 3
```shell
python libero/lifelong/eval_skill_chain.py \
--seed 10000 --device_id 0 \
--model_path_folder "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/experiments/libero_90/skill_policies_without_wrist_camera_view/Sequential/BCRNNPolicy_seed10000/all"
```

