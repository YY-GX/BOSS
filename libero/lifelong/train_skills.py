import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import pprint
import time
import hydra
import numpy as np
import wandb
import yaml
import torch
import h5py
from easydict import EasyDict
from omegaconf import OmegaConf
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.policy_starter import PolicyStarter
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    """
    Description: main() function for training separate skills
    """


    """
    Preparation - configs / random seeds / ...
    """
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))
    # prepare configs
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)
    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())
    # control seed
    control_seed(cfg.seed)

    """
    Prepare datasets - demos + language embeddings
    """
    benchmark = get_benchmark(cfg.benchmark_name)(n_tasks=cfg.task_num_to_use)
    n_manip_tasks = benchmark.n_tasks

    if cfg.is_debug:
        n_manip_tasks = 2
        cfg.train.n_epochs = 2
        cfg.eval.n_eval = 10
        cfg.eval.max_steps = 20

    # prepare datasets from the benchmark
    manip_datasets = []
    descriptions = []
    shape_meta = None
    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=True,
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
            exit(1)
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        # maintain a list containing (lang, ds)
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    # prepare language embeddings
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # prepare demos
    datasets = [
        SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
    ]

    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    n_tasks = n_manip_tasks  # number of tasks
    print("\n=================== Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        print(f"        {benchmark.get_task(i).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")


    """
    Start training
    """
    # prepare experiment and update the config
    create_experiment_dir(cfg, version=cfg.version)
    cfg.shape_meta = shape_meta
    if cfg.use_wandb:
        wandb.init(project="libero", config=cfg)
        wandb.run.name = cfg.experiment_name
    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # loss confusion matrix
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # success confusion matrix
        "L_fwd": np.zeros((n_manip_tasks,)),  # loss AUC, how fast the agent learns
        "S_fwd": np.zeros((n_manip_tasks,)),  # success AUC, how fast the agent succeeds
    }
    succ_list = []
    for i in range(n_tasks):
        # Save the experiment config file, so we can resume or replay later
        with open(os.path.join(cfg.experiment_dir, f"config_task{i}.json"), "w") as f:
            json.dump(cfg, f, cls=NpEncoder, indent=4)
        policy_starter = safe_device(PolicyStarter(n_tasks, cfg), cfg.device)
        print(f"[info] start training on task {i}")
        policy_starter.train()
        t0 = time.time()
        s_fwd, l_fwd = policy_starter.learn_one_task(
            datasets[i], None, policy_starter, i, benchmark, result_summary, cfg
        )
        result_summary["S_fwd"][i] = s_fwd
        result_summary["L_fwd"][i] = l_fwd

        t1 = time.time()

        """
        Start evaluation
        """
        # evalute on all seen tasks at the end of learning each task
        if cfg.eval.eval:
            policy_starter.eval()
            print("=========== Start Evaluation (Rollouts) ===========")
            L = evaluate_loss(cfg, policy_starter, benchmark, datasets[: i + 1])
            t2 = time.time()
            # rollout the policy here
            S = evaluate_success(
                cfg=cfg,
                algo=policy_starter,
                benchmark=benchmark,
                task_ids=[i],
                result_summary=result_summary if cfg.eval.save_sim_states else None,
                video_folder=os.path.join(cfg.experiment_dir, f"task{i}_videos")
            )
            print(f">> Success Rate: {S[0]}")
            wandb.log({f"task{i}/success_rate": S[0], "epoch": 0})
            succ_list.append(S[0])
            with open(os.path.join(cfg.experiment_dir, f"succ_list.npy"), 'wb') as f:
                np.save(f, np.array(succ_list))
            t3 = time.time()
            result_summary["L_conf_mat"][i][: i + 1] = L
            result_summary["S_conf_mat"][i][: i + 1] = S
            if cfg.use_wandb:
                wandb.run.summary["success_confusion_matrix"] = result_summary[
                    "S_conf_mat"
                ]
                wandb.run.summary["loss_confusion_matrix"] = result_summary[
                    "L_conf_mat"
                ]
                wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
                wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
            print(
                f"[info] train time (min) {(t1-t0)/60:.1f} "
                + f"eval loss time {(t2-t1)/60:.1f} "
                + f"eval success time {(t3-t2)/60:.1f}"
            )
            print(("[Task %2d loss ] " + " %4.2f |" * (i + 1)) % (i, *L))
            torch.save(
                result_summary, os.path.join(cfg.experiment_dir, f"result_task{i}.pt")
            )

    with open(os.path.join(cfg.experiment_dir, f"succ_list.npy"), 'wb') as f:
        np.save(f, np.array(succ_list))

    print("[info] Finished learning\n")
    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
