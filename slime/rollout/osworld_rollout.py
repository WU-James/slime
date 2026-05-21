"""OSWorld HTTP rollout: task JSON paths -> OSWorld /rollout -> traj -> slime Sample."""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import Namespace
from typing import Any

import httpx

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_PROCESSOR = None
_TOKENIZER = None
_SAMPLE_PRINTED = False

EXAMPLE_FILE_PREFIX = "examples/"


def _osworld_rollout_url() -> str:
    return os.environ.get("OSWORLD_ROLLOUT_URL", "http://127.0.0.1:18081/rollout")


def _osworld_root() -> str:
    root = os.environ.get("OSWORLD_ROOT")
    if not root:
        raise ValueError("Set OSWORLD_ROOT to the OSWorld repository path.")
    return os.path.abspath(root)


def _osworld_test_config_base_dir() -> str:
    root = _osworld_root()
    base = os.environ.get("OSWORLD_TEST_CONFIG_BASE_DIR") or os.path.join(root, "evaluation_examples")
    return os.path.abspath(base)


def _osworld_http_timeout() -> float:
    return float(os.environ.get("OSWORLD_ROLLOUT_TIMEOUT", "7200"))


def _osworld_rollout_max_retries() -> int:
    return int(os.environ.get("OSWORLD_ROLLOUT_MAX_RETRIES", "3"))


def get_example_file_from_sample(sample: Sample) -> str:
    if isinstance(sample.metadata, dict) and sample.metadata.get("example_file"):
        return str(sample.metadata["example_file"]).strip()
    if isinstance(sample.prompt, str) and sample.prompt.strip():
        return sample.prompt.strip()
    raise ValueError(f"Sample has no example_file in metadata or prompt: {sample}")


def absolute_to_osworld_example_file(path: str, test_config_base_dir: str) -> str:
    """Convert dataset path to OSWorld file_list entry (relative to test_config_base_dir)."""
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith(EXAMPLE_FILE_PREFIX) and normalized.endswith(".json"):
        return normalized

    base = os.path.abspath(test_config_base_dir)
    abs_path = os.path.abspath(normalized)
    if abs_path.startswith(base + os.sep):
        rel = os.path.relpath(abs_path, base).replace("\\", "/")
        if rel.startswith(EXAMPLE_FILE_PREFIX) and rel.endswith(".json"):
            return rel

    raise ValueError(
        f"Cannot convert path to OSWorld example_file (expected under {base} as "
        f"{EXAMPLE_FILE_PREFIX}<domain>/<id>.json): {path}"
    )


def post_osworld_rollout(
    args: Namespace,
    file_list: list[str],
    *,
    n: int | None = None,
) -> dict[str, Any]:
    """POST /rollout on the OSWorld server. file_list entries must be relative example paths."""
    url = _osworld_rollout_url()
    batch_size = len(file_list)
    n = n if n is not None else args.n_samples_per_prompt
    payload = {"batch_size": batch_size, "n": n, "file_list": file_list}
    timeout = _osworld_http_timeout()
    max_retries = _osworld_rollout_max_retries()

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload)
            if response.status_code == 409:
                wait_s = min(30, 5 * (attempt + 1))
                logger.warning("OSWorld server busy (409), retry in %ss", wait_s)
                time.sleep(wait_s)
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "completed":
                raise RuntimeError(f"OSWorld rollout failed: {data}")
            return data
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            wait_s = min(30, 5 * (attempt + 1))
            logger.warning("OSWorld rollout request failed (%s), retry in %ss", exc, wait_s)
            time.sleep(wait_s)
    raise RuntimeError(f"OSWorld rollout failed after {max_retries} attempts: {last_error}") from last_error


def _get_processor_and_tokenizer(args: Namespace):
    global _PROCESSOR, _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if _PROCESSOR is None:
        _PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    return _PROCESSOR, _TOKENIZER


def traj_paths_to_sample_groups(
    args: Namespace,
    prompt_groups: list[list[Sample]],
    traj_paths: list[str],
) -> list[list[Sample]]:
    """Map OSWorld traj_paths (length = num_tasks * n) back to slime sample groups."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from examples.osworld.traj_to_sample import traj_to_sample

    processor, tokenizer = _get_processor_and_tokenizer(args)
    osworld_root = _osworld_root()
    n = args.n_samples_per_prompt
    expected = len(prompt_groups) * n
    if len(traj_paths) != expected:
        raise ValueError(f"Expected {expected} traj paths, got {len(traj_paths)}")

    result_groups: list[list[Sample]] = []
    for group_idx, group in enumerate(prompt_groups):
        new_group: list[Sample] = []
        for sample_idx in range(n):
            traj_path = traj_paths[group_idx * n + sample_idx]
            if not os.path.isabs(traj_path):
                traj_path = os.path.join(osworld_root, traj_path)
            sample = traj_to_sample(
                traj_path,
                osworld_root=osworld_root,
                processor=processor,
                tokenizer=tokenizer,
                build_multimodal_train_inputs=True,
            )
            template = group[sample_idx]
            sample.group_index = template.group_index
            sample.index = template.index
            sample.label = template.label
            sample.metadata = {**(template.metadata or {}), **sample.metadata}
            new_group.append(sample)
        result_groups.append(new_group)
    return result_groups


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Fetch task paths from dataset, call OSWorld /rollout, load traj JSON as Samples."""
    global _SAMPLE_PRINTED

    if evaluation:
        return RolloutFnEvalOutput(data={})

    assert args.rollout_global_dataset
    test_config_base_dir = _osworld_test_config_base_dir()

    prompt_groups = data_source.get_samples(args.rollout_batch_size)
    file_list = [
        absolute_to_osworld_example_file(get_example_file_from_sample(group[0]), test_config_base_dir)
        for group in prompt_groups
    ]

    logger.info(
        "OSWorld rollout %s: batch_size=%s n=%s files=%s",
        rollout_id,
        len(file_list),
        args.n_samples_per_prompt,
        file_list,
    )
    response = post_osworld_rollout(args, file_list)
    traj_paths: list[str] = list(response["traj_paths"])
    logger.info(
        "OSWorld rollout %s done: rollout_idx=%s num_traj=%s",
        rollout_id,
        response.get("rollout_idx"),
        len(traj_paths),
    )

    data_groups = traj_paths_to_sample_groups(args, prompt_groups, traj_paths)

    if not _SAMPLE_PRINTED and data_groups:
        sample = data_groups[0][0]
        logger.info(
            "OSWorld rollout example: example_file=%s tokens=%s response_length=%s reward=%s",
            sample.metadata.get("example_file"),
            len(sample.tokens),
            sample.response_length,
            sample.reward,
        )
        _SAMPLE_PRINTED = True

    metrics = {
        "osworld/rollout_idx": response.get("rollout_idx"),
        "osworld/num_traj": len(traj_paths),
        "osworld/mean_reward": (
            sum(s.reward for g in data_groups for s in g if isinstance(s.reward, (int, float)))
            / max(sum(1 for g in data_groups for s in g if isinstance(s.reward, (int, float))), 1)
        ),
    }
    return RolloutFnTrainOutput(samples=data_groups, metrics=metrics)
