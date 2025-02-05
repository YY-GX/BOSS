import abc
import os
import glob
import random
import torch
import json

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.boss_task_map import boss_task_map\

"""
Create gloabl vars
"""
BENCHMARK_MAPPING = {}

"""
Base class
"""

class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, n_tasks=None):
        self.task_embs = None
        self.task_indexes = selected_task_indexes[self.name]
        self.n_tasks = n_tasks

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        print(f"[INFO] Using task orders {self.task_indexes}")
        self.tasks = [tasks[i] for i in self.task_indexes]

        if self.n_tasks:
            self.n_tasks = self.n_tasks
        else:
            # if n_tasks set to None, it means to use all tasks
            self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]


    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


"""
Tool classes/methods
"""

class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


def create_reverse_mapping(mapping):
    """Create a reverse mapping from values to keys."""
    reverse_mapping = {}
    for key, values in mapping.items():
        for value in values:
            reverse_mapping.setdefault(value, []).append(key)
    return reverse_mapping

def find_keys_by_value(mapping, target_value):
    """Find all keys associated with a given value in the mapping."""
    reverse_mapping = create_reverse_mapping(mapping)
    return reverse_mapping.get(target_value, [])

def grab_language_from_filename(x, is_modified=False):
    if is_modified:
        if "SCENE10" in x:
            language = " ".join(x.split("SCENE")[-1][3:].split("_"))
        else:
            language = " ".join(x.split("SCENE")[-1][2:].split("_"))
    elif x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]





"""
Register
"""

@register_benchmark
class BOSS_44(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "boss_44"
        self._make_benchmark()

@register_benchmark
class CHALLENGE_1(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "ch1"
        self._make_benchmark()


@register_benchmark
class CHALLENGE_2_2(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "ch2_2_modifications"
        self._make_benchmark()


@register_benchmark
class CHALLENGE_2_3(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "ch2_3_modifications"
        self._make_benchmark()



@register_benchmark
class FACTOR_1(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "factor_1"
        self._make_benchmark()


@register_benchmark
class FACTOR_2(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "factor_2"
        self._make_benchmark()


@register_benchmark
class DATA_AUGMENTATION(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "data_augmentation"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, n_tasks=None):
        super().__init__(n_tasks=n_tasks)
        self.name = "libero_90"
        self._make_benchmark()


"""
Create task_maps
"""
boss_suites = [
    "boss_44",
    "ch1",
    "ch2_2_modifications",
    "ch2_3_modifications",
    "factor_1",
    "factor_2",
    "data_augmentation",
    "libero_90",
]
task_maps = {}
max_len = 0
keep_language_unchanged = True
for boss_suite in boss_suites:
    task_maps[boss_suite] = {}

    for task in boss_task_map[boss_suite]:
        if (boss_suite == "boss_44") or (boss_suite == "libero_90"):
            # keep language unchanged - extract language directly based on bddl file name
            language = grab_language_from_filename(task + ".bddl", is_modified=False)
        else:
            # use original task's language
            mapping_pth = f"./libero/mappings/{boss_suite}.json"
            with open(mapping_pth, 'r') as json_file:
                mapping = json.load(json_file)
            task_ori = find_keys_by_value(mapping, task + ".bddl")[0]
            language = grab_language_from_filename(task_ori + ".bddl", is_modified=True)

        task_maps[boss_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=boss_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )



"""
Created Tasks Indexes
"""
selected_task_indexes = {
    "boss_44": [i for i in range(0, 44)],
    "ch1": [i for i in range(0, 44)],
    "ch2_2_modifications": [i for i in range(0, 44)],
    "ch2_3_modifications": [i for i in range(0, 44)],
    "ch3": [
        [2, 3, 5],
        [3, 2, 5],
        [6, 7, 10],
        [6, 8, 10],
        [19, 18, 16],
        [19, 16, 18],
        [17, 16, 20],
        [20, 17, 16],
        [32, 35, 36],
        [34, 35, 36],
    ],
    "factor_1": [i for i in range(0, 34)],
    "factor_2": [i for i in range(0, 20)],
    "data_augmentation": list(range(0, 1727)),
    "libero_90": [i for i in range(0, 90)],
}





# bl3_all_task_order = list(range(0, 1727))
#
# task_orders = [
#     # train skills (0 ~ 3)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
#     [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # transformer
#     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # rnn
#     [37, 38, 39, 40, 41, 42, 43],  # vilt
#     # eval_modified_env (4)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
#     # eval ori env (5)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     # my method - 1: eval modified env (with midify_back) (6)
#     [1, 11, 18],
#     # eval long horizon tasks (7 ~ 16)
#     [2, 3, 5],
#     [3, 2, 5],
#     [6, 7, 10],
#     [6, 8, 10],
#     [19, 18, 16],
#     [19, 16, 18],
#     [17, 16, 20],
#     [20, 17, 16],
#     [32, 35, 36],
#     [34, 35, 36],
#     # speed up training (17 ~ 20)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
#     [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
#     [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
#     # ablation-1 (21)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
#     # ablation-2 (22)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     # bl3_all (23)
#     bl3_all_task_order,
#     # diffusion policy - 1 (24)
#     [4],
#     [3],
#     [0],
#     [0, 1, 2, 3],
#     # real libero 90 (28)
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
# ]




