#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Reshape tools/datasets/cp8sp50k_custom_dataset_text_matched_token_ids.jsonl
# (one {"prompt": "..."} object per line) into the ShareGPT conversation
# schema inference-perf's `data.type: shareGPT` loader expects, since
# inference-perf has no dataset type that reads a bare prompt-only JSONL
# directly (see inference_perf/config/datagen/config.py's DataGenType enum).
#
# Each output record gets a synthetic second ("gpt") turn so the record has
# the >=2 turns hf_sharegpt_datagen.py requires, and so that -- with
# api.type: completion in inference-perf-config.yaml -- its token count
# becomes the request's max_tokens (inference_perf/datagen/
# hf_sharegpt_datagen.py:get_completion_data), letting us control the
# generated output length for this decode benchmark. The filler is only an
# approximation of --target-output-tokens words-as-tokens: inference-perf
# tokenizes it with the real model tokenizer at run time, not this script.
#
# Usage: convert_dataset_to_sharegpt.py <input_jsonl> <output_json> [--target-output-tokens N]
import argparse
import json
import sys

# Cycled (not repeated) so common BPE tokenizers don't collapse the filler
# into fewer, longer tokens than --target-output-tokens words implies.
FILLER_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "runs",
    "near", "river", "under", "bright", "morning", "sun", "while", "birds",
    "sing", "softly", "among", "tall", "green", "trees", "beside", "old",
    "stone", "bridge", "where", "children", "play", "every", "single", "day",
]


def build_filler(target_output_tokens: int) -> str:
    words = [FILLER_WORDS[i % len(FILLER_WORDS)] for i in range(target_output_tokens)]
    return " ".join(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_jsonl", help="Path to the source prompt-only JSONL dataset")
    parser.add_argument("output_json", help="Path to write the ShareGPT-format JSONL to (must end in .json)")
    parser.add_argument(
        "--target-output-tokens",
        type=int,
        default=256,
        help="Approximate token count for the synthetic 'gpt' filler turn (default: 256)",
    )
    args = parser.parse_args()

    if not args.output_json.endswith(".json"):
        sys.exit(
            "output_json must end in .json -- inference_perf/datagen/hf_sharegpt_datagen.py "
            "only loads paths ending in .json (the content is still JSON-lines)."
        )

    filler = build_filler(args.target_output_tokens)

    count = 0
    with open(args.input_jsonl, encoding="utf-8") as src, open(args.output_json, "w", encoding="utf-8") as dst:
        for line_num, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" not in row:
                sys.exit(f"{args.input_jsonl}:{line_num}: missing required 'prompt' field")
            record = {
                "id": count,
                "conversations": [
                    {"from": "human", "value": row["prompt"]},
                    {"from": "gpt", "value": filler},
                ],
            }
            dst.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} ShareGPT-format records to {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
