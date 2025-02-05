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
from libero.lifelong.algos import get_algo_class, get_algo_list
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


# Used when need to train based on my customized demos
def split_dataset(h5_location):
    # Open the HDF5 file
    with h5py.File(h5_location, 'r') as f:
        # Get the keys from the 'data' group
        data_keys = list(f['data'].keys())

        # Sort the keys to maintain order
        data_keys.sort()

        # Split the data into 40 for training and 10 for evaluation
        train_keys = data_keys[:40]
        eval_keys = data_keys[40:]

        # Determine the new file names
        base_name = os.path.splitext(os.path.basename(h5_location))[0]
        folder = os.path.dirname(h5_location)
        train_file_name = os.path.join(folder, f"{base_name}_train.h5")
        eval_file_name = os.path.join(folder, f"{base_name}_eval.h5")

        # Save the training data
        with h5py.File(train_file_name, 'w') as train_file:
            train_group = train_file.create_group('data')
            for key in train_keys:
                train_group.copy(f['data'][key], key)

        # Save the evaluation data
        with h5py.File(eval_file_name, 'w') as eval_file:
            eval_group = eval_file.create_group('data')
            for key in eval_keys:
                eval_group.copy(f['data'][key], key)

    print(f"Dataset split into {train_file_name} and {eval_file_name}.")



@hydra.main(config_path="../configs", config_name="config_no_ll", version_base=None)
def main(hydra_cfg):
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
    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())
    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())
    # control seed
    control_seed(cfg.seed)


    """
    Prepare datasets - demos + language embeddings
    """
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index, n_tasks_=cfg.task_num_to_use)
    n_manip_tasks = benchmark.n_tasks

    # prepare datasets from the benchmark
    manip_datasets = []
    manip_datasets_eval = []
    descriptions = []
    shape_meta = None
    if cfg.is_split:
        h5_file_location = os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(0)
                )
        h5_folder = os.path.dirname(h5_file_location)
        if not os.path.exists(os.path.join(
            h5_folder, f"{os.path.splitext(os.path.basename(h5_file_location))[0]}_train.h5"
        )):
            split_dataset(h5_file_location)

    for i in range(n_manip_tasks):
        # currently we assume tasks from same benchmark have the same shape_meta
        try:
            # yy: seq_length seems to be an important hyper-param
            if cfg.is_split:
                # yy: /home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/../datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
                task_i_dataset, shape_meta = get_dataset(
                    dataset_path=os.path.join(
                        h5_folder, f"{os.path.splitext(os.path.basename(h5_file_location))[0]}_train.h5"
                    ),
                    obs_modality=cfg.data.obs.modality,
                    # initialize_obs_utils=(i == 0),  # yy: ori, but in my case, everytime is a new restart
                    initialize_obs_utils=True,
                    seq_len=cfg.data.seq_len,
                )
                task_i_dataset_eval, shape_meta_eval = get_dataset(
                    dataset_path=os.path.join(
                        h5_folder, f"{os.path.splitext(os.path.basename(h5_file_location))[0]}_eval.h5"
                    ),
                    obs_modality=cfg.data.obs.modality,
                    # initialize_obs_utils=(i == 0),  # yy: ori, but in my case, everytime is a new restart
                    initialize_obs_utils=True,
                    seq_len=cfg.data.seq_len,
                )
            else:
                # yy: /home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/../datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
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
        # yy: they maintain a list containing (lang, ds)
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
        if cfg.is_split:
            manip_datasets_eval.append(task_i_dataset_eval)

    # prepare language embeddings
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    # prepare demos
    datasets = [
        SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
    ]
    if cfg.is_split:
        datasets_eval = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets_eval, task_embs)
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
        algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
        print(f"[info] start training on task {i}")
        algo.train()
        t0 = time.time()

        if cfg.is_split:
            s_fwd, l_fwd = algo.learn_one_task_no_ll(
                datasets[i], datasets_eval[i], algo, i, benchmark, result_summary, cfg
            )
        else:
            s_fwd, l_fwd = algo.learn_one_task_no_ll(
                datasets[i], None, algo, i, benchmark, result_summary, cfg
            )
        result_summary["S_fwd"][i] = s_fwd
        result_summary["L_fwd"][i] = l_fwd

        t1 = time.time()

        """
        Start evaluation
        """
        # evalute on all seen tasks at the end of learning each task
        if cfg.eval.eval:
            algo.eval()
            print("=========== Start Evaluation (Rollouts) ===========")
            if cfg.is_split:
                L = evaluate_loss(cfg, algo, benchmark, datasets_eval[: i + 1])
            else:
                L = evaluate_loss(cfg, algo, benchmark, datasets[: i + 1])
            t2 = time.time()
            # rollout the policy here
            S = evaluate_success(
                cfg=cfg,
                algo=algo,
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
