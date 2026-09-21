#!/usr/bin/env python3
"""Wire BOSS into a local OpenVLA checkout.

Copies the BOSS evaluation scripts into <openvla>/experiments/robot/libero/ and
registers the `libero44` and `libero_bl3_all` RLDS datasets that BOSS fine-tunes
on. Safe to re-run: every step is idempotent.

    python integrations/openvla/install.py [--openvla-dir openvla] [--check]
"""

import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent

# copied verbatim into <openvla>/experiments/robot/libero/
SCRIPTS = [
    "eval_openvla_ch1_ch2.py",
    "eval_openvla_ch3.py",
    "eval_openvla_speed_up.py",
    "libero_utils.py",            # BOSS version: adds get_libero_subproc_env
    "regenerate_libero_dataset.py",
]

# appended to the OXE registries so `--dataset_name libero44` resolves
DATASET_CONFIG = '''    "{name}": {{
        "image_obs_keys": {{"primary": "image", "secondary": None, "wrist": None}},
        "depth_obs_keys": {{"primary": None, "secondary": None, "wrist": None}},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    }},
'''
DATASET_TRANSFORM = '    "{name}": libero_dataset_transform,\n'
DATASETS = ["libero44", "libero_bl3_all"]

ANCHOR_CONFIG = '    "libero_10_no_noops": {'
ANCHOR_TRANSFORM = '    "libero_10_no_noops": libero_dataset_transform,'


def insert_before_anchor(path, anchor, blocks, check):
    """Insert each block before `anchor`, skipping blocks already present."""
    text = path.read_text(encoding="utf-8")
    if anchor not in text:
        raise SystemExit(
            f"[error] could not find the LIBERO registry anchor in {path}.\n"
            f"        Expected a line starting with: {anchor.strip()}\n"
            f"        OpenVLA's layout may have changed; register the datasets by hand."
        )
    missing = [b for name, b in blocks if f'"{name}"' not in text]
    if not missing:
        return False
    if check:
        return True
    text = text.replace(anchor, "".join(missing) + anchor, 1)
    path.write_text(text, encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openvla-dir", default="openvla",
                    help="Path to your OpenVLA checkout (default: ./openvla)")
    ap.add_argument("--check", action="store_true",
                    help="Report what would change without writing anything.")
    args = ap.parse_args()

    root = pathlib.Path(args.openvla_dir).resolve()
    libero_dir = root / "experiments" / "robot" / "libero"
    configs = root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "configs.py"
    transforms = root / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "transforms.py"

    for probe in (libero_dir, configs.parent):
        if not probe.is_dir():
            raise SystemExit(
                f"[error] {root} does not look like an OpenVLA checkout ({probe} missing).\n"
                f"        git clone https://github.com/openvla/openvla {args.openvla_dir}"
            )

    changed = []
    for name in SCRIPTS:
        src, dst = HERE / name, libero_dir / name
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        changed.append(f"copy {name} -> {dst}")
        if not args.check:
            shutil.copy2(src, dst)

    if insert_before_anchor(
            configs, ANCHOR_CONFIG,
            [(n, DATASET_CONFIG.format(name=n)) for n in DATASETS], args.check):
        changed.append(f"register {DATASETS} in {configs}")
    if insert_before_anchor(
            transforms, ANCHOR_TRANSFORM,
            [(n, DATASET_TRANSFORM.format(name=n)) for n in DATASETS], args.check):
        changed.append(f"register {DATASETS} in {transforms}")

    if not changed:
        print("Already installed; nothing to do.")
        return 0
    verb = "Would apply" if args.check else "Applied"
    print(f"{verb} {len(changed)} change(s):")
    for c in changed:
        print("  -", c)
    if args.check:
        return 1
    print("\nDone. Run the BOSS OpenVLA evaluations from inside the OpenVLA checkout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
