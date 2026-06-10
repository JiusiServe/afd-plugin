# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared GSM8K lm-eval integration helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def _run_lm_eval(
    base_url: str,
    model_name: str,
    *,
    output_path: str,
    num_fewshot: int | None = None,
    batch_size: int | None = None,
    max_tokens: int = 512,
    tokenizer: str | None = None,
    tasks_dir: str | None = None,
    limit: int | None = None,
) -> dict:
    """Run lm-eval against the AFD attention server and return results dict."""
    tokenizer_arg = f",tokenizer={tokenizer}" if tokenizer else ""
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        (
            f"model={model_name},"
            f"base_url={base_url}/v1/completions,"
            f"max_tokens={max_tokens},"
            f"tokenized_requests=False"
            f"{tokenizer_arg}"
        ),
    ]
    # When tasks_dir is given, load a local task config (e.g. a gsm8k.yaml that
    # points at pre-staged data files) instead of the built-in task that would
    # try to download the dataset from HuggingFace.
    if tasks_dir:
        cmd.extend(["--include_path", tasks_dir])
    cmd.extend(
        [
            "--tasks",
            "gsm8k",
            "--output_path",
            output_path,
            "--log_samples",
        ]
    )
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if batch_size is not None:
        cmd.extend(["--batch_size", str(batch_size)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"\n[lm-eval] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("USE_MODELSCOPE_HUB", "0")
    if tasks_dir:
        # Local task dir => fully offline data; do not let lm-eval/datasets phone home.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
    env["PYTHONUNBUFFERED"] = "1"
    # Stream lm-eval output live (pytest -s surfaces it) instead of capturing.
    # capture_output=True swallows everything until exit, which makes a slow run
    # indistinguishable from a deadlock — a real footgun on NPU eager-mode runs.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    deadline = time.monotonic() + 7200  # 2h max
    stdout_lines: list[str] = []
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line:
            stdout_lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
        elif proc.poll() is not None:
            break
        elif time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError("lm-eval exceeded 7200s budget")
    proc.wait()
    stdout_text = "".join(stdout_lines)

    if proc.returncode != 0:
        raise RuntimeError(
            f"lm-eval exited with code {proc.returncode}:\n{stdout_text[-3000:]}",
        )

    # Parse results: prefer results.json (lm-eval writes <output_path>/results.json),
    # but search the tree in case the exact layout differs between versions.
    op = Path(output_path)
    results_file = None
    if op.exists():
        if (op / "results.json").exists():
            results_file = op / "results.json"
        elif op.is_file():
            results_file = op
        else:
            hits = list(op.rglob("results.json"))
            results_file = hits[0] if hits else None
    if results_file is not None:
        with open(results_file) as f:
            return json.load(f)
    return _parse_lm_eval_stdout(stdout_text)


def _parse_lm_eval_stdout(stdout: str) -> dict:
    """Parse lm-eval results from stdout.

    Prefers a trailing JSON block (some versions print one). Falls back to the
    pipe-delimited results table lm-eval always prints, e.g. a strict-match row
    like: | strict-match | 5 | exact_match | 0.33 | +/- | 0.0473 |. Returns a
    dict shaped like results.json for _extract_gsm8k_accuracy.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    m = re.search(
        r"strict-match.*?exact_match[^0-9.\-+]*([0-9]*\.?[0-9]+)",
        stdout,
        re.DOTALL,
    )
    strict_val = float(m.group(1)) if m else None
    m2 = re.search(r"exact_match[^0-9.\-+]*([0-9]*\.?[0-9]+)", stdout, re.DOTALL)
    flex_val = float(m2.group(1)) if m2 else None
    if strict_val is None and flex_val is None:
        raise RuntimeError("Could not parse lm-eval results from stdout")
    gsm8k = {}
    if strict_val is not None:
        gsm8k["exact_match,strict-match"] = strict_val
    if flex_val is not None:
        gsm8k["exact_match,flexible-extract"] = flex_val
    gsm8k["exact_match"] = strict_val if strict_val is not None else flex_val
    return {"results": {"gsm8k": gsm8k}}


def _extract_gsm8k_accuracy(results: dict) -> float:
    """Extract the GSM8K exact_match,strict-match score from lm-eval output."""
    # Navigate the nested results structure
    # results["results"]["gsm8k"]["exact_match,strict-match"]
    task_results = results.get("results", results)

    gsm8k = task_results.get("gsm8k", task_results)
    for key in ("exact_match,strict-match", "exact_match"):
        if key in gsm8k:
            return float(gsm8k[key])

    raise KeyError(
        f"Could not find GSM8K accuracy in results. "
        f"Available keys: {list(gsm8k.keys())}",
    )
