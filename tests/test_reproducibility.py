"""Regression tests for the reproducibility bugs fixed in this repository.

Fast by design: none of these build a MuJoCo environment or load a checkpoint,
so they run in a couple of seconds and are safe to put in CI.

    pytest tests/
"""

import ast
import os
import pathlib
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
    """Covers every folder whose code we ship."""
    absolute = re.compile(r"[\"'](/mnt/|/home/|/playpen)")
    offenders = []
    roots = [os.path.join(REPO_ROOT, d) for d in ("libero", "scripts", "RAMG", "integrations")]
    for walk_root in roots:
      for root, dirs, files in os.walk(walk_root):
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



def _is_max_steps_override_guard(test):
    """Match exactly `args.max_steps is not None`, not any test mentioning max_steps."""
    return (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.left, ast.Attribute)
        and test.left.attr == "max_steps"
        and isinstance(test.left.value, ast.Name)
        and test.left.value.id == "args"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _parse(script):
    with open(os.path.join(REPO_ROOT, script), encoding="utf-8") as f:
        return ast.parse(f.read())


@pytest.mark.parametrize("script", [
    "libero/lifelong/eval_skill_chain.py",
    "libero/lifelong/eval_skills_affected_by_oss.py",
    "libero/lifelong/eval_skills_unaffected_by_oss.py",
])
def test_max_steps_override_guards_only_itself(script):
    """An indentation slip here silently pulls following statements into the branch."""
    tree = _parse(script)
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not _is_max_steps_override_guard(node.test):
            continue
        found += 1
        assert len(node.body) == 1, (
            f"{script}: the --max_steps branch guards {len(node.body)} statements, expected 1"
        )
    assert found == 1, f"{script}: expected exactly one --max_steps guard, found {found}"


def test_chain_eval_collects_every_sub_task_config():
    """cfg_ls drives how many sub-envs the chain gets; it must not sit behind a flag."""
    tree = _parse("libero/lifelong/eval_skill_chain.py")
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "append"
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id in {"cfg_ls", "task_ls", "algo_ls", "init_states_ls"}):
                raise AssertionError(
                    f"{inner.func.value.id}.append is inside a conditional; the skill chain "
                    f"would silently lose sub-tasks"
                )


def test_no_unguarded_wandb_calls():
    """use_wandb=false must not crash training: wandb.log needs wandb.init first."""
    offenders = []
    for path in sorted((pathlib.Path(REPO_ROOT) / "libero" / "lifelong").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        class Visitor(ast.NodeVisitor):
            def __init__(self):
                self.guards = []

            def visit_If(self, node):
                self.guards.append(ast.dump(node.test))
                self.generic_visit(node)
                self.guards.pop()

            def visit_Call(self, node):
                root = node.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if (isinstance(root, ast.Name) and root.id == "wandb"
                        and not any("use_wandb" in g for g in self.guards)):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
                self.generic_visit(node)

        Visitor().visit(tree)
    assert not offenders, "wandb calls not guarded by cfg.use_wandb: " + ", ".join(offenders)


def test_setup_py_actually_finds_the_packages():
    """Without libero/__init__.py, find_packages() returns nothing and
    `pip install -e .` installs an empty distribution."""
    from setuptools import find_packages

    packages = [p for p in find_packages(where=REPO_ROOT) if p.startswith("libero")]
    assert "libero" in packages, f"top-level libero package not found: {packages}"
    for expected in ["libero.libero", "libero.lifelong", "libero.libero.benchmark"]:
        assert expected in packages, f"{expected} missing from {packages}"


def test_environment_yml_and_requirements_agree():
    """environment.yml must defer to requirements.txt rather than re-pin versions."""
    import yaml

    env = yaml.safe_load(open(os.path.join(REPO_ROOT, "environment.yml")))
    pip_section = next((d["pip"] for d in env["dependencies"] if isinstance(d, dict)), [])
    assert pip_section == ["-r requirements.txt"], pip_section

    reqs = [
        line.strip()
        for line in open(os.path.join(REPO_ROOT, "requirements.txt"))
        if line.strip() and not line.startswith("#")
    ]
    assert len(reqs) > 20, reqs
    # every requirement is exactly pinned, so the environment is reproducible
    unpinned = [r for r in reqs if "==" not in r]
    assert not unpinned, f"unpinned requirements: {unpinned}"


@pytest.mark.parametrize("script", [
    "libero/lifelong/eval_skills_affected_by_oss.py",
    "libero/lifelong/eval_skills_unaffected_by_oss.py",
])
def test_debug_mode_truncates_the_task_loop(script):
    """n_tasks truncates task_embs; the loop must be truncated with it."""
    with open(os.path.join(REPO_ROOT, script), encoding="utf-8") as f:
        source = f.read()
    assert "benchmark.task_indexes[:n_tasks]" in source, (
        f"{script} iterates all task_indexes while n_tasks may be smaller, so "
        f"benchmark.get_task_emb() runs off the end of task_embs"
    )


def test_debug_mode_task_embeddings_stay_in_range():
    """The behaviour the above guards: one task in, one task out."""
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark("boss_44")(n_tasks=1)
    n_tasks = benchmark.n_tasks
    task_ids = benchmark.task_indexes[:n_tasks]
    benchmark.set_task_embs(torch.randn(n_tasks, 768))

    assert len(task_ids) == 1
    for idx in range(len(task_ids)):
        assert benchmark.get_task_emb(idx).shape == (768,)


def test_dataset_download_uses_a_live_source():
    """LIBERO's utexas.box.com links all 403; the downloader must not use them."""
    from libero.libero.utils import download_utils

    source = open(download_utils.__file__, encoding="utf-8").read()
    # the hostname may still appear in a comment; what must be gone is any URL
    assert "https://utexas.box.com" not in source, (
        "downloader still points at the dead Box links"
    )
    assert download_utils.HF_DATASET_REPO == "yifengzhu-hf/LIBERO-datasets"
    assert download_utils.DATASET_SUBSETS["libero_100"] == ["libero_10", "libero_90"]
    assert "libero_90" in download_utils.DATASET_SUBSETS


def test_demo_replay_tolerates_extra_degrees_of_freedom():
    """A RAMG modification can add objects, growing the env's qpos/qvel beyond
    what the demonstration stored. robosuite then raises ValueError, which used
    to be worked around by hand-editing site-packages."""
    tree = _parse("scripts/DemoProcessor.py")
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        calls = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_state_from_flattened"
        ]
        if not calls:
            continue
        handled = [
            h.type.id for h in node.handlers
            if isinstance(h.type, ast.Name)
        ]
        assert "ValueError" in handled, f"handlers are {handled}"
        guarded = True
    assert guarded, "set_state_from_flattened is not guarded against the dimension mismatch"
