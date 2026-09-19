"""Regression tests for the reproducibility bugs fixed in this repository.

Fast by design: none of these build a MuJoCo environment or load a checkpoint,
so they run in a couple of seconds and are safe to put in CI.

    pytest tests/
"""

import os
import re

import numpy as np
import pytest
import torch

from libero.libero.benchmark import MAPPINGS_FOLDER, boss_task_map
from libero.libero.envs.env_wrapper import SequentialEnv
from libero.lifelong.utils import control_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _RecordingEnv:
    def __init__(self):
        self.seeds = []

    def seed(self, seed):
        self.seeds.append(seed)
        np.random.seed(seed)  # what ControlEnv -> bddl_base_domain.seed does


def test_sequential_env_seed_reaches_every_sub_env():
    """`seed = np.random.seed(seed)` used to forward None to every sub-env."""
    chain = SequentialEnv.__new__(SequentialEnv)
    chain.env_ls = [_RecordingEnv() for _ in range(3)]

    chain.seed(10000)

    for env in chain.env_ls:
        assert env.seeds == [10000], f"sub-env received {env.seeds}, expected [10000]"


def test_sequential_env_seed_leaves_rng_deterministic():
    """The old code ended on np.random.seed(None), re-seeding from OS entropy."""
    draws = []
    for _ in range(3):
        chain = SequentialEnv.__new__(SequentialEnv)
        chain.env_ls = [_RecordingEnv() for _ in range(3)]
        chain.seed(10000)
        draws.append(float(np.random.rand()))

    assert len(set(draws)) == 1, f"global RNG not deterministic after seed(): {draws}"


def test_control_seed_makes_torch_sampling_reproducible():
    """The BC policies sample actions from a GMM head, so eval needs a torch seed."""
    def draw():
        return torch.randn(8).tolist()

    control_seed(10000)
    first = draw()
    control_seed(10000)
    second = draw()
    control_seed(10001)
    other = draw()

    assert first == second
    assert first != other


def test_mappings_folder_is_inside_this_checkout():
    """Guards against re-introducing an absolute path to someone's machine."""
    assert MAPPINGS_FOLDER.startswith(REPO_ROOT), MAPPINGS_FOLDER
    for suite in ["ch1", "ch2_2_modifications", "ch2_3_modifications"]:
        assert os.path.exists(os.path.join(MAPPINGS_FOLDER, f"{suite}.json"))


def test_no_hardcoded_home_paths_in_library_code():
    absolute = re.compile(r"[\"'](/mnt/|/home/|/playpen)")
    offenders = []
    for root, dirs, files in os.walk(os.path.join(REPO_ROOT, "libero")):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", "bddl_files", "init_files", "assets"}]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if absolute.search(line):
                        offenders.append(f"{os.path.relpath(path, REPO_ROOT)}:{lineno}")
    assert not offenders, "hardcoded machine paths: " + ", ".join(offenders)


def test_dataset_prep_keeps_exactly_the_boss_44_demos():
    """The old hand-written allow-list dropped 7 required demos and kept 9 extras."""
    script = os.path.join(REPO_ROOT, "scripts", "form_boss_44_dataset.py")
    with open(script, encoding="utf-8") as f:
        source = f.read()
    assert 'boss_task_map["boss_44"]' in source, (
        "the keep-list must be derived from the benchmark definition, not hand-written"
    )
    assert len(boss_task_map["boss_44"]) == 44


@pytest.mark.parametrize("script", [
    "libero/lifelong/eval_skill_chain.py",
    "libero/lifelong/eval_skills_affected_by_oss.py",
    "libero/lifelong/eval_skills_unaffected_by_oss.py",
])
def test_eval_scripts_seed_themselves(script):
    with open(os.path.join(REPO_ROOT, script), encoding="utf-8") as f:
        source = f.read()
    assert "control_seed(args.seed)" in source, f"{script} does not seed the run"


@pytest.mark.parametrize("script", [
    "libero/lifelong/eval_skills_affected_by_oss.py",
    "libero/lifelong/eval_skills_unaffected_by_oss.py",
])
def test_is_debug_defaults_to_off(script):
    with open(os.path.join(REPO_ROOT, script), encoding="utf-8") as f:
        source = f.read()
    block = source.split('"--is_debug"')[1].split(")")[0]
    assert "default=0" in block, f"{script} still defaults to debug mode"
