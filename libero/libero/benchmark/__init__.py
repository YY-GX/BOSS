import abc
import os
import glob
import random
import torch
import json

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from libero.libero.benchmark.yy_suite_task_map import yy_task_map

BENCHMARK_MAPPING = {}


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

class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(x, is_yy=False):
    if is_yy:
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


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_90_full"
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    problem_folder = libero_suite if libero_suite != "libero_90_full" else "libero_90"

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=problem_folder,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )


# yy: add my task
yy_suites = [
    # "yy_try",
    "modified_libero",
    "single_step",
    "ablation_1",
    "ablation_2",
    "multi_step_2",
    "multi_step_3",
    "bl3_all"
]
# yy: if you wanna the task description the same as the original one, set True here.
keep_language_unchanged = True
for yy_suite in yy_suites:
    task_maps[yy_suite] = {}

    for task in yy_task_map[yy_suite]:
        if keep_language_unchanged:
            mapping_pth = f"/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/mappings/{yy_suite}.json"
            with open(mapping_pth, 'r') as json_file:
                mapping = json.load(json_file)
            # if yy_suite == 'ablation_1':
            #     print(mapping)
            #     print(find_keys_by_value(mapping, task + ".bddl"))
            task_ori = find_keys_by_value(mapping, task + ".bddl")[0]
            language = grab_language_from_filename(task_ori + ".bddl", is_yy=True)
        else:
            language = grab_language_from_filename(task + ".bddl", is_yy=True)
        task_maps[yy_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=yy_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )


bl3_all_task_order = list(range(0, 1727))

task_orders = [
    # train skills (0 ~ 3)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # transformer
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # rnn
    [37, 38, 39, 40, 41, 42, 43],  # vilt
    # eval_modified_env (4)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    # eval ori env (5)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    # my method - 1: eval modified env (with midify_back) (6)
    [1, 11, 18],
    # eval long horizon tasks (7 ~ 16)
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
    # speed up training (17 ~ 20)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    # ablation-1 (21)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
    # ablation-2 (22)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    # bl3_all (23)
    bl3_all_task_order,
    # diffusion policy - 1 (24)
    [4],
    [3],
    [0],
    [0, 1, 2, 3],
    # real libero 90 (28)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
]



class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0, n_tasks_=None):
        self.task_embs = None
        self.task_order_index = task_order_index
        self.n_tasks_ = n_tasks_
        print(f"[INFO] Benchmark task order index: {self.task_order_index}")

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        if (self.name == "yy_try"):
            self.tasks = tasks
        else:
            print(f"[info] using task orders {task_orders[self.task_order_index]}")
            self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
        # yy: set 1 for just traininig 1 task
        if self.n_tasks_:
            self.n_tasks = self.n_tasks_
        else:
            # if n_tasks_ set to None, it means to use all tasks
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

@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    # FIXED: remember to change 28 back to 0 later
    def __init__(self, task_order_index=0, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        # yy: I comment this
        # assert (
        #     task_order_index == 0
        # ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_90_Full(Benchmark):
    def __init__(self, task_order_index=28, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "libero_90_full"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()


@register_benchmark
class yy_try(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=1):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "yy_try"
        self._make_benchmark()


@register_benchmark
class modified_libero(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=1):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "modified_libero"
        self._make_benchmark()


@register_benchmark
class single_step(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "single_step"
        self._make_benchmark()


@register_benchmark
class ablation_1(Benchmark):
    def __init__(self, task_order_index=21, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "ablation_1"
        self._make_benchmark()


@register_benchmark
class ablation_2(Benchmark):
    def __init__(self, task_order_index=22, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "ablation_2"
        self._make_benchmark()


@register_benchmark
class multi_step_2(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "multi_step_2"
        self._make_benchmark()


@register_benchmark
class multi_step_3(Benchmark):
    def __init__(self, task_order_index=0, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "multi_step_3"
        self._make_benchmark()

@register_benchmark
class bl3_all(Benchmark):
    def __init__(self, task_order_index=23, n_tasks_=None):
        super().__init__(task_order_index=task_order_index, n_tasks_=n_tasks_)
        self.name = "bl3_all"
        self._make_benchmark()

