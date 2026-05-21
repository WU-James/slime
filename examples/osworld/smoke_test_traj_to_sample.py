#!/usr/bin/env python3
"""Smoke test: OSWorld traj JSON -> slime Sample -> train_data contract."""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on sys.path for `slime` and local imports.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.osworld.traj_to_sample import (  # noqa: E402
    infer_osworld_root,
    load_traj_paths_as_samples,
    samples_to_train_data_dict,
    validate_sample_against_traj,
    validate_traj_for_sample,
    load_traj_json,
    traj_to_sample,
)


def _default_traj_paths(osworld_root: str) -> list[str]:
    traj_dir = os.path.join(osworld_root, "traj", "0")
    if not os.path.isdir(traj_dir):
        return []
    return sorted(
        os.path.join(traj_dir, name)
        for name in os.listdir(traj_dir)
        if name.endswith(".json")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traj_paths",
        nargs="*",
        help="Traj JSON paths (default: all JSON under $OSWORLD_ROOT/traj/0/)",
    )
    parser.add_argument(
        "--osworld-root",
        default=os.environ.get("OSWORLD_ROOT"),
        help="OSWorld repo root for resolving traj_images/* paths",
    )
    parser.add_argument(
        "--hf-checkpoint",
        default=os.environ.get("HF_CHECKPOINT"),
        help="HF model path for processor alignment check (optional)",
    )
    parser.add_argument(
        "--skip-processor",
        action="store_true",
        help="Only validate traj schema and Sample fields (no processor run)",
    )
    args = parser.parse_args()

    osworld_root = args.osworld_root
    traj_paths = list(args.traj_paths)
    if not traj_paths:
        if not osworld_root:
            guessed = infer_osworld_root(
                os.path.join(_REPO_ROOT, "..", "OSWorld", "traj", "0", "dummy.json")
            )
            osworld_root = guessed or os.path.join(os.path.dirname(_REPO_ROOT), "OSWorld")
        traj_paths = _default_traj_paths(osworld_root)

    if not traj_paths:
        print("No traj JSON files found. Pass paths or set OSWORLD_ROOT.", file=sys.stderr)
        return 1

    print(f"osworld_root={osworld_root}")
    print(f"checking {len(traj_paths)} traj file(s)\n")

    processor = None
    tokenizer = None
    build_mm = not args.skip_processor
    if build_mm:
        if not args.hf_checkpoint:
            print("HF_CHECKPOINT not set; running without processor (--skip-processor).")
            build_mm = False
        else:
            from slime.utils.processing_utils import load_processor, load_tokenizer

            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
            if processor is None:
                print(f"Failed to load processor from {args.hf_checkpoint}", file=sys.stderr)
                return 1

    failed = 0
    samples = []
    for path in traj_paths:
        print("=" * 60)
        print(path)
        traj = load_traj_json(path)
        traj_errors = validate_traj_for_sample(traj)
        if traj_errors:
            failed += 1
            print("  TRAJ INVALID:")
            for err in traj_errors:
                print(f"    - {err}")
            continue
        print("  traj schema: OK")

        try:
            sample = traj_to_sample(
                path,
                osworld_root=osworld_root,
                processor=processor,
                tokenizer=tokenizer,
                build_multimodal_train_inputs=build_mm,
            )
        except Exception as exc:
            failed += 1
            print(f"  traj_to_sample FAILED: {exc}")
            continue

        sample_errors = validate_sample_against_traj(sample, traj)
        if sample_errors:
            failed += 1
            print("  sample mismatch:")
            for err in sample_errors:
                print(f"    - {err}")
            continue

        trainable = sum(sample.loss_mask or [])
        print(f"  Sample: tokens={len(sample.tokens)} response_length={sample.response_length}")
        print(f"  loss_mask trainable tokens={trainable}")
        print(f"  multimodal_train_inputs={'set' if sample.multimodal_train_inputs else 'none'}")
        print("  traj_to_sample: OK")
        samples.append(sample)

    if samples:
        data = samples_to_train_data_dict(samples)
        print("\n" + "=" * 60)
        print("train_data contract (RolloutManager-compatible):")
        print(f"  num_samples={len(samples)}")
        print(f"  total_lengths={data['total_lengths']}")
        for sample, tl, rl, lm in zip(
            samples, data["total_lengths"], data["response_lengths"], data["loss_masks"], strict=True
        ):
            assert tl == len(sample.tokens)
            assert rl == sample.response_length
            assert len(lm) == rl
            assert tl - rl >= 0
        print("  train_data: OK")

    print("\n" + ("PASS" if failed == 0 else f"FAIL ({failed} file(s))"))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
