"""
Download functionalities adapted from Mandlekar et. al.: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""
import os
import time
from tqdm import tqdm
from termcolor import colored
from pathlib import Path
import zipfile
import io
import urllib.request
import shutil

from libero.libero import get_libero_path

DIR = os.path.dirname(__file__)

# LIBERO's original utexas.box.com links all return 403; upstream now publishes
# the same demonstrations on the HuggingFace Hub.
HF_DATASET_REPO = "yifengzhu-hf/LIBERO-datasets"

# "libero_100" is the union of the two long/short task suites, kept as an alias
# for the old API. BOSS itself only needs libero_90.
DATASET_SUBSETS = {
    "libero_object": ["libero_object"],
    "libero_goal": ["libero_goal"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero_100": ["libero_10", "libero_90"],
}


class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.
    Args:
        url (str): url string
    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    # request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, check_overwrite=True, is_zipfile=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.
    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.
    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    user_response = None
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n\n"
        )
        # assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    if user_response is None or user_response.lower() in {"yes", "y"}:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=fname
        ) as t:
            urllib.request.urlretrieve(
                url, filename=file_to_write, reporthook=t.update_to
            )
    if is_zipfile:
        with zipfile.ZipFile(file_to_write, "r") as archive:
            archive.extractall(path=download_dir)
        if os.path.isfile(file_to_write):
            os.remove(file_to_write)


def libero_dataset_download(datasets="all", download_dir=None, check_overwrite=True):
    """Download LIBERO demonstrations from the HuggingFace Hub.

    Args:
        datasets (str): a key of DATASET_SUBSETS, or "all".
        download_dir (str, optional): target directory. Defaults to the
            configured datasets path.
        check_overwrite (bool): unused, kept for API compatibility --
            huggingface_hub skips files that are already present.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "huggingface_hub is required to download the datasets: pip install huggingface_hub"
        ) from exc

    valid = sorted(DATASET_SUBSETS) + ["all"]
    if datasets not in valid:
        raise ValueError(f"Unknown dataset {datasets!r}; choose one of {valid}")

    if download_dir is None:
        download_dir = get_libero_path("datasets")
    os.makedirs(download_dir, exist_ok=True)

    wanted = sorted(DATASET_SUBSETS) if datasets == "all" else [datasets]
    subsets = sorted({s for name in wanted for s in DATASET_SUBSETS[name]})

    print(f"Downloading {subsets} from {HF_DATASET_REPO} into {download_dir}")
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=[f"{s}/*" for s in subsets],
    )


def check_libero_dataset(download_dir=None):
    """Check the integrity of the downloaded datasets.

    Args:
        download_dir (str, optional): The path where datasets are stored. Defaults to None, using the default path.

    Returns:
        bool: True if the datasets are successfully downloaded, False otherwise.
    """
    if download_dir is None:
        download_dir = get_libero_path("datasets")
    check_result = True
    for dataset_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_90",
    ]:
        info_str = ""
        dataset_status = False
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            count = 0
            for path in Path(dataset_dir).glob("*.hdf5"):
                count += 1
            if (count == 10 and dataset_name != "libero_90") or (
                count == 90 and dataset_name == "libero_90"
            ):
                dataset_status = True
                info_str = colored(
                    f"[X] Dataset {dataset_name} is complete", "green", attrs=["bold"]
                )
            else:
                colored(
                    f"[?] Dataset {dataset_name} is not downloaded completely",
                    "yellow",
                    attrs=["bold"],
                )
        else:
            info_str = colored(
                f"[ ] Dataset {dataset_name} not found!!!", "red", attrs=["bold"]
            )

        print(info_str)
        check_result = check_result and dataset_status
    return check_result
