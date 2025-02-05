import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import multiprocessing
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
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

import h5py


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
    # preprocessing
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # print configs to terminal
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # control seed
    control_seed(cfg.seed)

    # prepare lifelong learning
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index, n_tasks_=1)
    # yy: libero/libero/benchmark/__init__.py - line 123 is responsible for the n_manip_tasks
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

    # yy: For debug
    # i = 0
    # # yy: /home/yygx/UNC_Research/pkgs_simu/LIBERO/libero/libero/../datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
    # task_i_dataset, shape_meta = get_dataset(
    #     dataset_path=os.path.join(
    #         h5_folder, f"{os.path.splitext(os.path.basename(h5_file_location))[0]}_train.h5"
    #     ),
    #     obs_modality=cfg.data.obs.modality,
    #     # initialize_obs_utils=(i == 0),  # yy: ori, but in my case, everytime is a new restart
    #     initialize_obs_utils=True,
    #     seq_len=cfg.data.seq_len,
    # )
    # task_i_dataset_eval, shape_meta_eval = get_dataset(
    #     dataset_path=os.path.join(
    #         h5_folder, f"{os.path.splitext(os.path.basename(h5_file_location))[0]}_eval.h5"
    #     ),
    #     obs_modality=cfg.data.obs.modality,
    #     # initialize_obs_utils=(i == 0),  # yy: ori, but in my case, everytime is a new restart
    #     initialize_obs_utils=True,
    #     seq_len=cfg.data.seq_len,
    # )
    # exit(0)

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
                    # initialize_obs_utils=(i == 0),  # yy: ori, but in my case, everytime is a new restart
                    initialize_obs_utils=True,
                    seq_len=cfg.data.seq_len,
                )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # add language to the vision dataset, hence we call vl_dataset
        task_description = benchmark.get_task(i).language
        # yy: they maintain a list containing (lang, ds)
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)
        if cfg.is_split:
            manip_datasets_eval.append(task_i_dataset_eval)

    # yy: this task_embs seem to be the language embeddings
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    gsz = cfg.data.task_group_size
    if gsz == 1:  # each manipulation task is its own lifelong learning task
        datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
        ]
        if cfg.is_split:
            datasets_eval = [
                SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets_eval, task_embs)
            ]
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:  # group gsz manipulation tasks into a lifelong task, currently not used
        assert (
            n_manip_tasks % gsz == 0
        ), f"[error] task_group_size does not divide n_tasks"
        datasets = []
        n_demos = []
        n_sequences = []
        for i in range(0, n_manip_tasks, gsz):
            dataset = GroupedTaskDataset(
                manip_datasets[i : i + gsz], task_embs[i : i + gsz]
            )
            datasets.append(dataset)
            n_demos.extend([x.n_demos for x in dataset.sequence_datasets])
            n_sequences.extend(
                [x.total_num_sequences for x in dataset.sequence_datasets]
            )



    n_tasks = n_manip_tasks // gsz  # number of lifelong learning tasks
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        for j in range(gsz):
            print(f"        {benchmark.get_task(i*gsz+j).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")


    # yy: my analyses
    # print(type(datasets[0]))
    # print(len(datasets[0]))  # 3828
    # print(datasets[0].n_demos)  # 50
    # print(datasets[1].n_demos)  # 50
    # print(len(datasets))  # 2
    # print(descriptions)  # ['close the top drawer of the cabinet',
    # 'close the top drawer of the cabinet and put the black bowl on top of it']
    # print(n_tasks)
    # exit(0)


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

    # yy: default is false
    if cfg.eval.save_sim_states:
        # for saving the evaluate simulation states, so we can replay them later
        for k in range(n_manip_tasks):
            for p in range(k + 1):  # for testing task p when the agent learns to task k
                result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
            for e in range(
                cfg.train.n_epochs + 1
            ):  # for testing task k at the e-th epoch when the agent learns on task k
                if e % cfg.eval.eval_every == 0:
                    result_summary[f"k{k}_e{e//cfg.eval.eval_every}"] = [
                        [] for _ in range(cfg.eval.n_eval)
                    ]

    # define lifelong algorithm
    # yy: default is sequential
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
    # yy: default is ""
    if cfg.pretrain_model_path != "":  # load a pretrained model if there is any
        try:
            algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
        except:
            print(
                f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
            )
            sys.exit(0)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    GFLOPs, MParams = compute_flops(algo, datasets[0], cfg)
    print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # save the experiment config file, so we can resume or replay later
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    if cfg.lifelong.algo == "Multitask":

        algo.train()
        s_fwd, l_fwd = algo.learn_all_tasks(datasets, benchmark, result_summary)
        result_summary["L_fwd"][-1] = l_fwd
        result_summary["S_fwd"][-1] = s_fwd

        # evalute on all seen tasks at the end if eval.eval is true
        if cfg.eval.eval:
            # yy: where obtain eval loss
            L = evaluate_loss(cfg, algo, benchmark, datasets)
            S = evaluate_success(
                cfg=cfg,
                algo=algo,
                benchmark=benchmark,
                task_ids=list(range(n_manip_tasks)),
                result_summary=result_summary if cfg.eval.save_sim_states else None,
            )

            result_summary["L_conf_mat"][-1] = L
            result_summary["S_conf_mat"][-1] = S

            if cfg.use_wandb:
                wandb.run.summary["success_confusion_matrix"] = result_summary[
                    "S_conf_mat"
                ]
                wandb.run.summary["loss_confusion_matrix"] = result_summary[
                    "L_conf_mat"
                ]
                wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
                wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
                wandb.run.summary.update()

            print(("[All task loss ] " + " %4.2f |" * n_tasks) % tuple(L))
            print(("[All task succ.] " + " %4.2f |" * n_tasks) % tuple(S))

            torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))
    else:
        # for i in range(n_tasks-1, -1, -1):
        for i in range(n_tasks):
            algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
            print(f"[info] start training on task {i}")
            algo.train()

            t0 = time.time()
            # yy: learn_one_task_no_ll() is the function for training, modified by me
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

            # evalute on all seen tasks at the end of learning each task
            if cfg.eval.eval:
                print("=========== YY: Start Evaluation (Rollouts) ===========")
                if cfg.is_split:
                    L = evaluate_loss(cfg, algo, benchmark, datasets_eval[: i + 1])
                else:
                    L = evaluate_loss(cfg, algo, benchmark, datasets[: i + 1])
                t2 = time.time()
                # S = evaluate_success(
                #     cfg=cfg,
                #     algo=algo,
                #     benchmark=benchmark,
                #     task_ids=list(range((i + 1) * gsz)),
                #     result_summary=result_summary if cfg.eval.save_sim_states else None,
                # )
                S = evaluate_success(
                    cfg=cfg,
                    algo=algo,
                    benchmark=benchmark,
                    task_ids=[i],
                    result_summary=result_summary if cfg.eval.save_sim_states else None,
                )
                print(f">> Success Rate: {S}")
                wandb.log({f"task{i}/success_rate": S, "epoch": 0})
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
                    # wandb.run.summary.update()

                print(
                    f"[info] train time (min) {(t1-t0)/60:.1f} "
                    + f"eval loss time {(t2-t1)/60:.1f} "
                    + f"eval success time {(t3-t2)/60:.1f}"
                )
                print(("[Task %2d loss ] " + " %4.2f |" * (i + 1)) % (i, *L))
                # print(("[Task %2d succ.] " + " %4.2f |" * (i + 1)) % (i, *S))
                torch.save(
                    result_summary, os.path.join(cfg.experiment_dir, f"result.pt")
                )

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    # yy: I comment this
    # if multiprocessing.get_start_method(allow_none=True) != "spawn":
    #     multiprocessing.set_start_method("spawn", force=True)

    main()


