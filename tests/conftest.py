# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared pytest helpers for the AFD plugin test suite."""

from __future__ import annotations

import os
from pathlib import Path


def download_dataset(dataset_id: str, dataset_config: str | None = None) -> None:
    """Download/cache a Hugging Face dataset.

    Args:
        dataset_id: Dataset repo id, e.g. ``openai/gsm8k``.
        dataset_config: Optional dataset configuration name, e.g. ``main``.
    """
    from datasets import load_dataset

    if dataset_config is None:
        load_dataset(dataset_id)
    else:
        load_dataset(dataset_id, dataset_config)
    print(f"[e2e] Dataset ready: {dataset_id}", flush=True)


def download_model(repo_id: str) -> Path:
    """Download/cache a Hugging Face model repo and return its local path.

    Args:
        repo_id: Model repo id, e.g. ``deepseek-ai/DeepSeek-V2-Lite``.
    """
    # Must be set before importing huggingface_hub: Xet chunk caches can fill
    # the container root filesystem in CI even when HF_HOME is on a large volume.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download models; "
            "install it in the E2E environment",
        ) from exc

    print(f"[e2e] Downloading model {repo_id}", flush=True)
    model_path = Path(snapshot_download(repo_id=repo_id))
    print(f"[e2e] Model ready at {model_path}", flush=True)
    return model_path
