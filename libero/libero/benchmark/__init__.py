import abc
import os
import glob
import random
import torch
import json

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.boss_task_map import boss_task_map

"""
Create global vars
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
        if self.name in eval_ori_suites:
            tasks = sorted(tasks, key=lambda item: item.name)

        print(f"[INFO] Task indexes in current set: {self.task_indexes}")
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
        self.name = "boss_44"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()

@register_benchmark
class CH1(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "ch1"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()


@register_benchmark
class CH2_2_MODIFICATIONS(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "ch2_2_modifications"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()


@register_benchmark
class CH2_3_MODIFICATIONS(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "ch2_3_modifications"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()



@register_benchmark
class FACTOR_1(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "factor_1"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()


@register_benchmark
class FACTOR_2(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "factor_2"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()


@register_benchmark
class DATA_AUGMENTATION(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "data_augmentation"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, n_tasks=None):
        self.name = "libero_90"
        super().__init__(n_tasks=n_tasks)
        self._make_benchmark()


# @register_benchmark
# class G1(Benchmark):
#     def __init__(self, n_tasks=None):
#         self.name = "g1"
#         super().__init__(n_tasks=n_tasks)
#         self._make_benchmark()
#
#
# @register_benchmark
# class G2(Benchmark):
#     def __init__(self, n_tasks=None):
#         self.name = "g2"
#         super().__init__(n_tasks=n_tasks)
#         self._make_benchmark()
#
#
# @register_benchmark
# class G3(Benchmark):
#     def __init__(self, n_tasks=None):
#         self.name = "g3"
#         super().__init__(n_tasks=n_tasks)
#         self._make_benchmark()
#
#
# @register_benchmark
# class G4(Benchmark):
#     def __init__(self, n_tasks=None):
#         self.name = "g4"
#         super().__init__(n_tasks=n_tasks)
#         self._make_benchmark()


def create_benchmark_class(name):
    """Dynamically creates a Benchmark subclass with a given name."""
    return type(
        name,
        (Benchmark,),
        {
            "__init__": lambda self, n_tasks=None, name=name: (
                setattr(self, "name", name),
                super(self.__class__, self).__init__(n_tasks=n_tasks),
                self._make_benchmark(),
            )[0]
        }
    )

def register_benchmark_classes(prefix, start, end):
    """Generates and registers benchmark classes dynamically."""
    for i in range(start, end + 1):
        class_name = f"{prefix}{i}"
        globals()[class_name] = create_benchmark_class(class_name.lower())  # Assign dynamically
        register_benchmark(globals()[class_name])  # Register each class

# Generate and register CH3_1 to CH3_10
register_benchmark_classes("CH3_", 1, 10)

# Generate and register G1 to G8
register_benchmark_classes("G", 1, 8)

# Generate and register libero_local1 TODO
register_benchmark_classes("LIBERO_LOCAL", 1, 2)

# Generate and register ch1_libero_local1 TODO
register_benchmark_classes("CH1_LIBERO_LOCAL", 1, 2)


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
            # mapping_pth = f"./libero/mappings/{boss_suite}.json"
            mapping_pth = f"/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/mappings/{boss_suite}.json"
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


ch3_suites = [
    "ch3_1",
    "ch3_2",
    "ch3_3",
    "ch3_4",
    "ch3_5",
    "ch3_6",
    "ch3_7",
    "ch3_8",
    "ch3_9",
    "ch3_10",
]

for ch3_suite in ch3_suites:
    task_maps[ch3_suite] = task_maps["boss_44"].copy()


eval_ori_suites = [
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g6",
    "g7",
    "g8",
    "libero_local1",
]

for ori_suite in eval_ori_suites:
     task_maps[ori_suite] = task_maps["libero_90"].copy()


eval_ori_suites = [

    "libero_local2",  # TODO
]

for ori_suite in eval_ori_suites:
     task_maps[ori_suite] = task_maps["boss_44"].copy()


eval_ch1_suites = [
    "ch1_libero_local1",
    "ch1_libero_local2",
]

for ch1_suite in eval_ch1_suites:
     task_maps[ch1_suite] = task_maps["ch1"].copy()




"""
Created Tasks Indexes
"""
selected_task_indexes = {
    # "boss_44": [i for i in range(0, 44)],  # TODO: uncomment this
    # "boss_44": [i for i in range(33, 44)],
    "boss_44": [i for i in range(6, 44)],  # TODO: uncomment this
    "ch1": [i for i in range(0, 44)],
    "ch2_2_modifications": [i for i in range(0, 44)],
    "ch2_3_modifications": [i for i in range(0, 44)],
    "ch3_1": [2, 3, 5],
    "ch3_2": [3, 2, 5],
    "ch3_3": [6, 7, 10],
    "ch3_4": [6, 8, 10],
    "ch3_5": [19, 18, 16],
    "ch3_6": [19, 16, 18],
    "ch3_7": [17, 16, 20],
    "ch3_8": [20, 17, 16],
    "ch3_9": [32, 35, 36],
    "ch3_10": [34, 35, 36],
    "factor_1": [i for i in range(0, 34)],
    "factor_2": [i for i in range(0, 20)],
    "data_augmentation": list(range(0, 1727)),
    "libero_90": [i for i in range(0, 90)],
    "g1": [22, 13, 42, 7, 39, 62, 37, 84, 51, 31, 79, 18, 10, 52, 3, 40, 45, 64, 0, 72],
    "g2": [7, 19, 88, 29, 67, 86, 61, 63, 56, 25, 26, 64, 18, 83, 4, 60, 71, 15, 45, 66],
    "g3": [39, 22, 61, 84, 49, 89, 51, 47, 6, 85, 0, 82, 50, 26, 57, 23, 17, 74, 83, 55],
    "g4": [39, 49, 20, 61, 31, 89, 13, 6, 67, 1, 41, 57, 30, 80, 77, 5, 23, 54, 55, 38],
    "g5": [22, 13, 42, 7, 39, 40, 45, 64, 0, 72],
    "g6": [22, 13, 45, 64, 0],
    "g7": [0, 80, 34, 3, 10, 81, 21, 1, 19, 51],
    "g8": [81, 21, 1, 19, 51],
    "libero_local1": [2, 3, 4, 5, 9],
    "libero_local2": [7, 10, 15, 16, 17, 18, 19, 20, 23, 39],  # old: [6, 7, 10, 15, 17, 18, 19, 20, 23, 39] - 6 doesnt involve gripper open/close
    "ch1_libero_local1": [1, 4],  # correspond to [2, \3, \4, \5, 9] - [2, 9]
    "ch1_libero_local2": [7, 10, 15, 16, 17, 18, 19, 20, 23, 39],  # 16 is wrong, it doesn't involve gripper closing thing -> KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet
    # TODO
}


