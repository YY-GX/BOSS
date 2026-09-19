from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="boss-benchmark",
    packages=[package for package in find_packages() if package.startswith("libero")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description=(
        "BOSS: Benchmark for Observation Space Shift in Long-Horizon Task. "
        "Built on top of LIBERO (Liu et al., 2023)."
    ),
    author="Yue Yang, Linfeng Zhao, Mingyu Ding, Gedas Bertasius, Daniel Szafir",
    author_email="yygx@cs.unc.edu",
    url="https://boss-benchmark.github.io/",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
