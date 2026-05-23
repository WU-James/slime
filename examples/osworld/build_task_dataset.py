#!/usr/bin/env python3
"""Build slime prompt jsonl with absolute example_file paths from OSWorld evaluation_examples/examples/."""

from __future__ import annotations

import argparse
import json
import os


def collect_example_files(examples_dir: str) -> list[str]:
    examples_dir = os.path.abspath(examples_dir)
    if not os.path.isdir(examples_dir):
        raise FileNotFoundError(examples_dir)

    paths: list[str] = []
    for domain in sorted(os.listdir(examples_dir)):
        domain_dir = os.path.join(examples_dir, domain)
        if not os.path.isdir(domain_dir):
            continue
        for name in sorted(os.listdir(domain_dir)):
            if name.endswith(".json"):
                paths.append(os.path.abspath(os.path.join(domain_dir, name)))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-dir",
        default=None,
        help="Path to evaluation_examples/examples (default: <osworld-root>/evaluation_examples/examples)",
    )
    parser.add_argument(
        "--osworld-root",
        default=os.environ.get("OSWORLD_ROOT", "/home/yongjinwu/work/git/OSWorld"),
        help="OSWorld repository root",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output jsonl path (default: <osworld-root>/datasets/osworld_tasks.jsonl)",
    )
    args = parser.parse_args()

    osworld_root = os.path.abspath(args.osworld_root)
    examples_dir = args.examples_dir or os.path.join(osworld_root, "evaluation_examples", "examples")
    output = args.output or os.path.join(osworld_root, "datasets", "osworld_tasks.jsonl")

    records = [{"example_file": p} for p in collect_example_files(examples_dir)]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} tasks to {output}")
    print("example_file entries are absolute paths under evaluation_examples/examples/.")
    print("osworld_rollout converts them to relative paths for OSWorld /rollout file_list.")


if __name__ == "__main__":
    main()
