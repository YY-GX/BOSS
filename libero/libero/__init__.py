"""Path resolution for the BOSS benchmark assets, datasets and task files.

Paths live in ``<repo>/.boss/config.yaml``, which is generated on first import
and is deliberately *not* tracked by git: it holds absolute paths that are only
valid on the machine that created it. Set ``BOSS_CONFIG_PATH`` to keep it
elsewhere.
"""

import os

import yaml

# <repo>/libero/libero -- everything else is resolved relative to this, so the
# package works from any working directory and from any checkout.
_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PACKAGE_ROOT, os.pardir, os.pardir))

boss_config_path = os.environ.get("BOSS_CONFIG_PATH", os.path.join(_REPO_ROOT, ".boss"))
config_file = os.path.join(boss_config_path, "config.yaml")

_warned_missing = set()


def get_default_path_dict(custom_location=None):
    benchmark_root_path = _PACKAGE_ROOT if custom_location is None else custom_location
    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": os.path.join(benchmark_root_path, "bddl_files"),
        "init_states": os.path.join(benchmark_root_path, "init_files"),
        "datasets": os.path.join(os.path.dirname(benchmark_root_path), "datasets"),
        "assets": os.path.join(benchmark_root_path, "assets"),
    }


def _write_config(path_dict):
    os.makedirs(boss_config_path, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.safe_dump(path_dict, f, default_flow_style=False)
    return path_dict


def _load_config():
    """Return the path config, regenerating it if it belongs to another checkout."""
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f) or {}
        if config.get("benchmark_root") == _PACKAGE_ROOT:
            return config
        print(
            f"[BOSS] {config_file} points at {config.get('benchmark_root')!r}, "
            f"but this checkout lives at {_PACKAGE_ROOT!r}. Regenerating it."
        )
    else:
        print(f"[BOSS] Creating path config at {config_file}")
    return _write_config(get_default_path_dict())


def get_libero_path(query_key):
    config = _load_config()
    assert query_key in config, (
        f"Key {query_key} not found in config file {config_file}. "
        f"Available keys are: {list(config)}"
    )
    path = config[query_key]
    if not os.path.exists(path) and query_key not in _warned_missing:
        _warned_missing.add(query_key)
        print(f"[BOSS][Warning] {query_key} path {path} does not exist!")
    return path


def set_libero_default_path(custom_location=_PACKAGE_ROOT):
    """Re-point every path at ``custom_location`` and persist the result."""
    print(f"[BOSS] Rewriting {config_file} to use benchmark root {custom_location}")
    return _write_config(get_default_path_dict(custom_location))


_load_config()
