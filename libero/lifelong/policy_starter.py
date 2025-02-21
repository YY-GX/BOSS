import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *
import wandb

class PolicyStarter(nn.Module):
    """
    For skill training
    """

    def __init__(self, n_tasks, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_scale = cfg.train.loss_scale
        self.n_tasks = n_tasks
        if not hasattr(cfg, "experiment_dir"):
            create_experiment_dir(cfg)
            print(
                f"[info] Experiment directory not specified. Creating a default one: {cfg.experiment_dir}"
            )
        self.experiment_dir = cfg.experiment_dir
        self.algo = cfg.lifelong.algo

        self.policy = get_policy_class(cfg.policy.policy_type)(cfg, cfg.shape_meta)
        self.current_task = -1

    def end_task(self, dataset, task_id, benchmark, env=None):
        """
        What the algorithm does at the end of learning each lifelong task.
        """
        pass

    def start_task(self, task):
        """
        What the algorithm does at the beginning of learning each lifelong task.
        """
        self.current_task = task

        # initialize the optimizer and scheduler
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(), **self.cfg.train.optimizer.kwargs
        )

        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                T_max=self.cfg.train.n_epochs,
                **self.cfg.train.scheduler.kwargs,
            )

    def map_tensor_to_device(self, data):
        """Move data to the device specified by self.cfg.device."""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )


    # Optimization
    def observe(self, data):
        """
        How the algorithm learns on each data point.
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()


    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            loss = self.policy.compute_loss(data)
        return loss.item()


    # Train each skill policy
    def learn_one_task(self, dataset, datasets_eval, algo, task_id, benchmark, result_summary, cfg):

        self.start_task(task_id)

        # recover the corresponding manipulation task ids
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=self.cfg.train.num_workers > 0
        )

        # for evaluate how fast the agent learns on current task, this corresponds
        # to the area under success rate curve on the new task.
        cumulated_counter = 0.0
        successes = []
        losses = []

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        # start training
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if epoch > 0:  # update
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:  # just evaluate the zero-shot performance on 0-th epoch
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )
            wandb.log({f"task{task_id}/train_loss": training_loss, "epoch": epoch})

            if epoch % self.cfg.eval.eval_every == 0:
                t0 = time.time()
                if datasets_eval:
                    L = evaluate_loss(cfg, algo, benchmark, [datasets_eval])
                else:
                    L = evaluate_loss(cfg, algo, benchmark, [dataset])
                t1 = time.time()
                torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                print(
                    f"[info] Epoch: {epoch:3d} | eval loss: {training_loss:5.2f} | time: {(t1 - t0) / 60:4.2f}"
                )
                wandb.log({f"task{task_id}/eval_loss": L, "epoch": epoch})

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()


        # Do nothing in end_task()
        self.end_task(dataset, task_id, benchmark)

        # return the metrics regarding forward transfer
        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )

        return 0, 0


    def reset(self):
        self.policy.reset()
