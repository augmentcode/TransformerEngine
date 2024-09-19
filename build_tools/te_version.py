# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine version string."""
import os
from packaging.version import parse
from pathlib import Path
import subprocess


def te_version() -> str:
    """Transformer Engine version string

    Includes Git commit as local version, unless suppressed with
    NVTE_NO_LOCAL_VERSION environment variable.

    """
    root_path = Path(__file__).resolve().parent
    with open(root_path / "VERSION.txt", "r") as f:
        version = f.readline().strip()

    # [augment] Here is where we replace the git hash with our own versioning.
    # You can disable this behavior with NVTE_NO_AUGMENT_VERSION=1.
    if not int(os.getenv("NVTE_NO_AUGMENT_VERSION", "0")):
        # NOTE: we are assuming you are building for pytorch. TE cannot make this assumption in general.
        import torch
        torch_version = parse(torch.__version__)
        cuda_version = parse(torch.version.cuda)
        version_string = f".cu{cuda_version.major}{cuda_version.minor}.torch{torch_version.major}{torch_version.minor}"
        return version + "+augment" + version_string


    if not int(os.getenv("NVTE_NO_LOCAL_VERSION", "0")) and not bool(
        int(os.getenv("NVTE_RELEASE_BUILD", "0"))
    ):
        try:
            output = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                cwd=root_path,
                check=True,
                universal_newlines=True,
            )
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            commit = output.stdout.strip()
            version += f"+{commit}"
    return version
