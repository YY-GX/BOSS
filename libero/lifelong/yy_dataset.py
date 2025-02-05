import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
    ):
        super(SequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None  # HDF5 file handle, which will be opened lazily

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that need to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                obs_keys_in_memory = [
                    k for k in self.obs_keys if ObsUtils.key_is_obs_modality(k, "low_dim")
                ]
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs,
            )

            if self.hdf5_cache_mode == "all":
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [
                    self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))
                ]
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        # Do not close the file in the main process, as it should be opened in each worker process.
        # self.close_and_delete_hdf5_handle()

    def load_demo_info(self, filter_by_attribute=None, demos=None):
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [
                elem.decode("utf-8")
                for elem in np.array(
                    self.hdf5_file["mask/{}".format(filter_by_attribute)][:]
                )
            ]
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        self._index_to_demo_id = dict()
        self._demo_id_to_start_indices = dict()
        self._demo_id_to_demo_length = dict()

        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(
                self.hdf5_path, "r", swmr=self.hdf5_use_swmr, libver="latest"
            )
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        msg = (
            f"{self.__class__.__name__} (\n"
            f"\tpath={self.hdf5_path}\n"
            f"\tobs_keys={self.obs_keys}\n"
            f"\tseq_length={self.seq_length}\n"
            f"\tfilter_key={self.filter_by_attribute if self.filter_by_attribute else 'none'}\n"
            f"\tframe_stack={self.n_frame_stack}\n"
            f"\tpad_seq_length={self.pad_seq_length}\n"
            f"\tpad_frame_stack={self.pad_frame_stack}\n"
            f"\tgoal_mode={self.goal_mode if self.goal_mode else 'none'}\n"
            f"\tcache_mode={self.hdf5_cache_mode if self.hdf5_cache_mode else 'none'}\n"
            f"\tnum_demos={self.n_demos}\n"
            f"\tnum_sequences={self.total_num_sequences}\n"
            f")"
        )
        return msg

    def __len__(self):
        return self.total_num_sequences

    def load_dataset_in_memory(
        self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs
    ):
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {
                "num_samples": hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            }
            all_data[ep]["obs"] = {
                k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype("float32")
                for k in obs_keys
            }
            if load_next_obs:
                all_data[ep]["next_obs"] = {
                    k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype(
                        "float32"
                    )
                    for k in obs_keys
                }
            for k in dataset_keys:
                all_data[ep][k] = (
                    hdf5_file["data/{}/{}".format(ep, k)][()].astype("float32")
                    if k in hdf5_file["data/{}".format(ep)]
                    else np.zeros(
                        (all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32
                    )
                )
            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file[
                    "data/{}".format(ep)
                ].attrs["model_file"]
        return all_data

    def normalize_obs(self):
        def _compute_traj_stats(traj):
            stats = {}
            for k in traj:
                traj_array = np.array(traj[k])
                stats[k] = {
                    "n": traj_array.shape[0],
                    "mean": traj_array.mean(axis=0),
                    "sqdiff": ((traj_array - traj_array.mean(axis=0)) ** 2).sum(
                        axis=0
                    ),
                }
            return stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = (
                    traj_stats_a[k]["n"],
                    traj_stats_a[k]["mean"],
                    traj_stats_a[k]["sqdiff"],
                )
                n_b, avg_b, M2_b = (
                    traj_stats_b[k]["n"],
                    traj_stats_b[k]["mean"],
                    traj_stats_b[k]["sqdiff"],
                )
                n = n_a + n_b
                delta = avg_b - avg_a
                merged_stats[k] = {
                    "n": n,
                    "mean": avg_a + delta * (n_b / n),
                    "sqdiff": M2_a + M2_b + (delta**2) * (n_a * n_b / n),
                }
            return merged_stats

        def _compute_agg_stats_mean_std(agg_stats):
            mean_std_stats = {}
            for k in agg_stats:
                mean = agg_stats[k]["mean"]
                std = np.sqrt(agg_stats[k]["sqdiff"] / agg_stats[k]["n"])
                mean_std_stats[k] = (mean, std)
            return mean_std_stats

        with self.hdf5_file_opened():
            demo_ids = self.demos
            agg_stats = None
            for ep in demo_ids:
                ep_stats = _compute_traj_stats(
                    {
                        k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()]
                        for k in self.obs_keys
                    }
                )
                if agg_stats is None:
                    agg_stats = ep_stats
                else:
                    agg_stats = _aggregate_traj_stats(agg_stats, ep_stats)

            obs_normalization_stats = _compute_agg_stats_mean_std(agg_stats)
            return obs_normalization_stats

    def get_obs(self, demo, idx):
        obs = {
            k: self.hdf5_cache[demo]["obs"][k][idx]
            if self.hdf5_cache is not None
            else self.hdf5_file["data/{}/obs/{}".format(demo, k)][idx]
            for k in self.obs_keys
        }
        if self.hdf5_normalize_obs:
            obs = ObsUtils.normalize_obs(
                obs_dict=obs, obs_normalization_stats=self.obs_normalization_stats
            )
        return obs

    def get_sequence_from_demo(self, demo, start_idx, with_next_obs=True):
        end_idx = start_idx + self.seq_length - 1

        if self.hdf5_cache is not None:
            demo_length = self.hdf5_cache[demo]["attrs"]["num_samples"]
        else:
            demo_length = self.hdf5_file["data/{}".format(demo)].attrs["num_samples"]

        if self.pad_frame_stack:
            start_frame_idx = max(0, start_idx - (self.n_frame_stack - 1))
        else:
            start_frame_idx = start_idx
        frame_obs_indices = np.arange(start_frame_idx, start_idx + 1)

        if self.pad_seq_length:
            if end_idx >= demo_length:
                pad_len = end_idx - demo_length + 1
                end_idx = demo_length - 1
            else:
                pad_len = 0
        else:
            pad_len = 0
        obs_indices = np.arange(start_idx, end_idx + 1)

        frame_indices = []
        for t in obs_indices:
            start = t - self.n_frame_stack + 1
            end = t + 1
            indices = np.arange(max(start, 0), end)
            if start < 0 and self.pad_frame_stack:
                indices = np.pad(indices, (abs(start), 0), "edge")
            frame_indices.append(indices)

        seq_demo = demo
        obs = {
            k: np.stack(
                [
                    np.concatenate(
                        [
                            self.get_obs(seq_demo, f_idx)[k][None, ...]
                            for f_idx in frame_indices[i]
                        ],
                        axis=0,
                    )
                    for i in range(len(obs_indices))
                ],
                axis=0,
            )
            for k in self.obs_keys
        }

        pad_mask = None
        if self.get_pad_mask:
            pad_mask = np.ones(len(obs_indices), dtype=np.float32)
            pad_mask[-pad_len:] = 0

        seq_dict = dict(obs=obs, pad_mask=pad_mask)

        return seq_dict

    def get_item(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_idx = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]
        seq_start_idx = index - demo_start_idx
        seq_dict = self.get_sequence_from_demo(
            demo_id,
            start_idx=seq_start_idx,
            with_next_obs=self.load_next_obs,
        )

        if "actions" in self.dataset_keys:
            seq_dict["actions"] = np.zeros((self.seq_length, 1), dtype=np.float32)
            start = seq_start_idx
            end = seq_start_idx + self.seq_length
            seq_dict["actions"][: end - start] = (
                self.hdf5_cache[demo_id]["actions"][start:end]
                if self.hdf5_cache is not None
                else self.hdf5_file["data/{}/actions".format(demo_id)][start:end]
            )

        return TensorUtils.to_tensor(seq_dict, device="cpu", check=True)

    def __getitem__(self, index):
        if self.hdf5_cache_mode == "all":
            return deepcopy(self.getitem_cache[index])
        return self.get_item(index)
