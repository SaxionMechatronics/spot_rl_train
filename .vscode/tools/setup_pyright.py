# Copyright (c) 2026, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

# Copyright (c) 2026, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Update the extraPaths in pyproject.toml for Pyright/Pylance based on an IsaacLab installation path.

This script discovers all source packages under ``{isaaclab_path}/source/`` and writes them as
absolute paths into the ``[tool.pyright]`` ``extraPaths`` array in the root ``pyproject.toml``.
Run it once after cloning, or any time the IsaacLab installation path changes.

Usage:
    python3 .vscode/tools/setup_pyright.py --isaaclab_path /path/to/IsaacLab
"""

import argparse
import pathlib
import re
import sys

PROJECT_DIR = pathlib.Path(__file__).parents[2]
PYPROJECT_TOML = PROJECT_DIR / "pyproject.toml"


def find_isaaclab_source_paths(isaaclab_path: pathlib.Path) -> list[str]:
    """Return sorted absolute paths for every package directory under ``{isaaclab_path}/source/``."""
    source_dir = isaaclab_path / "source"
    if not source_dir.exists():
        print(f"[ERROR] Could not find source directory: {source_dir}")
        sys.exit(1)

    paths = sorted(str(entry) for entry in source_dir.iterdir() if entry.is_dir() and not entry.name.startswith("."))
    if not paths:
        print(f"[ERROR] No package directories found under: {source_dir}")
        sys.exit(1)

    return paths


def update_extra_paths(paths: list[str]) -> None:
    """Replace the extraPaths array in pyproject.toml with the given paths."""
    content = PYPROJECT_TOML.read_text()

    formatted = "[\n" + "".join(f'    "{p}",\n' for p in paths) + "]"
    updated, n = re.subn(
        r"extraPaths\s*=\s*\[.*?\]",
        f"extraPaths = {formatted}",
        content,
        flags=re.DOTALL,
    )

    if n == 0:
        print("[ERROR] Could not find 'extraPaths' in pyproject.toml. Is [tool.pyright] configured?")
        sys.exit(1)

    PYPROJECT_TOML.write_text(updated)
    print(f"Updated extraPaths in {PYPROJECT_TOML} with {len(paths)} path(s):")
    for p in paths:
        print(f"  {p}")


def main():
    parser = argparse.ArgumentParser(description="Update pyproject.toml extraPaths for IsaacLab.")
    parser.add_argument("--isaaclab_path", type=str, required=True, help="Absolute path to the IsaacLab root directory.")
    args = parser.parse_args()

    isaaclab_path = pathlib.Path(args.isaaclab_path).expanduser().resolve()
    if not isaaclab_path.exists():
        print(f"[ERROR] IsaacLab path does not exist: {isaaclab_path}")
        sys.exit(1)

    paths = find_isaaclab_source_paths(isaaclab_path)
    update_extra_paths(paths)


if __name__ == "__main__":
    main()
