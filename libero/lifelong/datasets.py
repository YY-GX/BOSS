import copy
import os.path

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from PIL import Image
from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import Dataset

"""
    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)


# Below ones are implemented by me - designed for BL3
import h5py
import pickle
import torch
from torch.utils.data import Dataset


def get_combined_dataset(
    dataset_path_ls,
    obs_modality,
    ratios_ls,
    only_success=False,
    succ_dict_path_ls=None,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):
    """
    Load and combine multiple datasets based on specified ratios.

    Parameters:
    - dataset_path_ls (list of str): List of paths to datasets to combine.
    - obs_modality (dict): Observation modalities to use.
    - ratios_ls (list of float): Ratios for combining datasets.
    - only_success (bool): If True, only use success cases from each dataset.
    - succ_dict_path_ls (list of str): List of paths to pickle files containing success and failure indices.
    - initialize_obs_utils (bool): Whether to initialize observation utilities.
    - seq_len (int): Sequence length for the datasets.
    - frame_stack (int): Frame stacking for the datasets.
    - filter_key (optional): Key to filter the datasets.
    - hdf5_cache_mode (str): HDF5 caching mode.
    - *args, **kwargs: Additional arguments passed to SequenceDataset.

    Returns:
    - Combined_Dataset: An instance of Combined_Dataset containing the datasets.
    - shape_meta: Metadata about the shapes of the datasets.
    """
    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list

    shape_meta_list = []
    datasets = []

    empty_ds_idx_ls = []
    for i, dataset_path in enumerate(dataset_path_ls):
        # yy: Jump is ds size is 0
        dataset_path = os.path.expanduser(dataset_path)
        f = h5py.File(dataset_path, "r")
        if len(list(f["data"].keys())) == 0:
            empty_ds_idx_ls.append(i)
            continue

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
        )
        shape_meta_list.append(shape_meta)

        # Create the dataset
        dataset = SequenceDataset(
            hdf5_path=dataset_path,
            obs_keys=shape_meta["all_obs_keys"],
            dataset_keys=["actions"],
            load_next_obs=False,
            frame_stack=frame_stack,
            seq_length=seq_len,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=hdf5_cache_mode,
            hdf5_use_swmr=False,
            hdf5_normalize_obs=None,
            filter_by_attribute=filter_key,
        )

        # Filter for success cases if required
        # Last element (i.e., Original dataset) will not be affected, only augmented datasets are affected
        if only_success and succ_dict_path_ls is not None and i < len(succ_dict_path_ls):
            if not os.path.exists(succ_dict_path_ls[i]):
                print(f"[WARNING] Whole dataset dropped because {succ_dict_path_ls[i]} doesn't exist!!")
                empty_ds_idx_ls.append(i)
                continue
            with open(succ_dict_path_ls[i], 'rb') as f:
                succ_dict = pickle.load(f)

            success_indices = succ_dict["success_idx"]
            # Filter the dataset to only include success cases
            sequence_ds_n_demos = dataset.n_demos
            sequence_ds_total_num_sequences = dataset.total_num_sequences
            dataset = torch.utils.data.Subset(dataset, success_indices)
            dataset.n_demos = sequence_ds_n_demos
            dataset.total_num_sequences = sequence_ds_total_num_sequences

        datasets.append(dataset)

    # Pop error cases ratio
    ratios_ls = [value for idx, value in enumerate(ratios_ls) if idx not in empty_ds_idx_ls]

    # Create the combined dataset with the specified ratios
    combined_dataset = Combined_Dataset(datasets, ratios_ls)

    return combined_dataset, shape_meta_list


class Combined_Dataset(Dataset):
    def __init__(self, datasets, ratios):
        """
        Combined_Dataset to combine multiple datasets with specific ratios.

        Parameters:
        - datasets (list of Dataset): List of datasets to combine.
        - ratios (list of float): Corresponding ratios for each dataset.
        """
        assert len(datasets) == len(ratios), "Datasets and ratios must have the same length"
        assert sum(ratios) > 0, "Sum of ratios must be greater than 0"

        self.datasets = datasets
        self.ratios = ratios
        self.total_length = self._calculate_total_length()

        self.n_demos = sum([ds.n_demos for ds in self.datasets])
        self.total_num_sequences = sum([ds.total_num_sequences for ds in self.datasets])

    def _calculate_total_length(self):
        """Calculate the total length of the combined dataset."""
        total_length = 0
        for dataset, ratio in zip(self.datasets, self.ratios):
            total_length += int(len(dataset) * ratio)  # Only use the specified ratio of each dataset
        return total_length

    def __len__(self):
        """Return the total length of the combined dataset."""
        return self.total_length

    def __getitem__(self, idx):
        """Retrieve an item from the combined dataset based on the index."""
        cumulative_lengths = []
        total_samples = 0

        # Calculate cumulative lengths based on ratios
        for dataset, ratio in zip(self.datasets, self.ratios):
            num_samples = int(len(dataset) * ratio)  # Number of samples to take from this dataset
            cumulative_lengths.append(total_samples + num_samples)
            total_samples += num_samples

        # Determine which dataset this index corresponds to
        dataset_idx = next(i for i, cumulative in enumerate(cumulative_lengths) if idx < cumulative)

        # Calculate the adjusted index for the selected dataset
        if dataset_idx == 0:
            adjusted_idx = idx  # Directly the requested index for the first dataset
        else:
            adjusted_idx = idx - cumulative_lengths[dataset_idx - 1]

        # Calculate the ratio-adjusted index
        ratio = self.ratios[dataset_idx]
        adjusted_idx = int(adjusted_idx / ratio)  # Scale the index back to the original dataset

        # Ensure adjusted_idx is within bounds
        if adjusted_idx < 0 or adjusted_idx >= len(self.datasets[dataset_idx]):
            raise IndexError(
                f"Adjusted index {adjusted_idx} is out of bounds for dataset {dataset_idx} with length {len(self.datasets[dataset_idx])}")

        return self.datasets[dataset_idx][adjusted_idx]




