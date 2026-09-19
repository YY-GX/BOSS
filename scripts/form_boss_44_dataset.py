"""Turn a downloaded LIBERO-90 demo folder into the 44-task `boss_44` dataset.

The set of tasks to keep is derived from ``boss_task_map["boss_44"]`` so that it
cannot drift from the benchmark definition. Files that are not part of boss_44
are moved aside rather than deleted; pass --delete to remove them instead.
"""

import argparse
import os
import shutil

from libero.libero import get_libero_path
from libero.libero.benchmark.boss_task_map import boss_task_map


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Folder holding the downloaded demos. Defaults to the 'datasets' path in .boss/config.yaml.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Name of the downloaded LIBERO-90 folder inside --dataset-dir. "
             "Defaults to the only sub-folder present.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete non-boss_44 demos instead of moving them to <dataset-dir>/libero_90_unused/.",
    )
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir or get_libero_path("datasets")
    target = os.path.join(dataset_dir, "boss_44")
    keep = {f"{task}_demo.hdf5" for task in boss_task_map["boss_44"]}

    if not os.path.isdir(target):
        source = args.source
        if source is None:
            subdirs = [
                d for d in sorted(os.listdir(dataset_dir))
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]
            if len(subdirs) != 1:
                raise SystemExit(
                    f"Expected exactly one sub-folder in {dataset_dir} to rename, found {subdirs}. "
                    f"Pass --source explicitly."
                )
            source = subdirs[0]
        os.rename(os.path.join(dataset_dir, source), target)
        print(f"Renamed {source} -> boss_44")

    present = {f for f in os.listdir(target) if f.endswith(".hdf5")}
    missing = sorted(keep - present)
    extra = sorted(present - keep)

    print(f"{target}: {len(present)} demos, {len(keep)} needed by boss_44")
    if missing:
        raise SystemExit(
            f"[error] {len(missing)} demos required by boss_44 are missing, e.g. {missing[:3]}. "
            f"Re-download LIBERO-90 before continuing."
        )
    if not extra:
        print("Nothing to do: the folder already holds exactly the boss_44 demos.")
        return

    action = "delete" if args.delete else "move aside"
    print(f"About to {action} {len(extra)} demo(s) that are not part of boss_44.")
    if not args.yes:
        if input("Continue? [y/N] ").strip().lower() != "y":
            raise SystemExit("Aborted.")

    unused_dir = os.path.join(dataset_dir, "libero_90_unused")
    if not args.delete:
        os.makedirs(unused_dir, exist_ok=True)
    for name in extra:
        path = os.path.join(target, name)
        if args.delete:
            os.remove(path)
        else:
            shutil.move(path, os.path.join(unused_dir, name))
    print(f"Done. boss_44 now holds {len(keep)} demos.")
    if not args.delete:
        print(f"The other {len(extra)} demo(s) were moved to {unused_dir}.")


if __name__ == "__main__":
    main()
