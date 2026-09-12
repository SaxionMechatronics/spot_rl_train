# Copyright (c) 2026, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""Installation script for the 'spot_rl_train' python package."""

import os

from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# Find all sub-packages (tasks, tasks.manager_based, etc.) and prepend the package name
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
_sub_packages = find_packages(where=EXTENSION_PATH)
ALL_PACKAGES = ["spot_rl_train"] + [f"spot_rl_train.{p}" for p in _sub_packages]

# Installation operation
setup(
    name="spot_rl_train",
    version="0.1.0",
    author="Kousheek Chakraborty",
    maintainer="Kousheek Chakraborty",
    description="Spot RL Training Extension for Isaac Lab",
    keywords=["extension", "isaaclab", "spot", "rl"],
    package_dir={"spot_rl_train": "."},
    packages=ALL_PACKAGES,
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
