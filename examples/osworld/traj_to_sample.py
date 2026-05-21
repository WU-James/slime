"""Load OSWorld rollout traj JSON files into slime Sample objects."""

from __future__ import annotations

import json
import os
from typing import Any

from slime.utils.types import Sample

_PROCESSOR_PROMPT_KEYS = frozenset({"input_ids", "attention_mask"})


def infer_osworld_root(traj_path: str | os.PathLike[str]) -> str | None:
    """Infer OSWorld repo root when traj lives under `<root>/traj/<idx>/*.json`."""
    path = os.path.abspath(traj_path)
    marker = f"{os.sep}traj{os.sep}"
    if marker in path:
        return path.split(marker, 1)[0]
    return None


def resolve_traj_asset_path(
    path: str,
    *,
    traj_path: str | os.PathLike[str],
    osworld_root: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve image paths stored in traj metadata (often relative to OSWorld cwd)."""
    if os.path.isabs(path) and os.path.isfile(path):
        return path

    traj_path = os.path.abspath(traj_path)
    candidates: list[str] = []
    if osworld_root:
        candidates.append(os.path.join(os.path.abspath(osworld_root), path))
    candidates.append(os.path.join(os.path.dirname(traj_path), path))
    candidates.append(os.path.join(os.path.dirname(traj_path), "..", path))

    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if os.path.isfile(normalized):
            return normalized

    raise FileNotFoundError(
        f"Could not resolve traj asset path '{path}' (traj={traj_path}, osworld_root={osworld_root})"
    )


def load_traj_json(traj_path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(traj_path, encoding="utf-8") as f:
        return json.load(f)


def _build_multimodal_inputs_from_image_paths(image_paths: list[str]) -> dict[str, Any]:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    return {"images": images, "videos": None}


def collapse_vision_tokens_for_processor(
    tokens: list[int],
    spans: list[list[int]],
    processor: Any,
) -> str:
    """Replace expanded vision spans with one placeholder triple each (for processor re-expand)."""
    hf_tokenizer = processor.tokenizer
    vision_start = hf_tokenizer.convert_tokens_to_ids("<|vision_start|>")
    image_pad = hf_tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_end = hf_tokenizer.convert_tokens_to_ids("<|vision_end|>")
    placeholder = [vision_start, image_pad, vision_end]

    collapsed: list[int] = []
    cursor = 0
    for start, end in spans:
        collapsed.extend(tokens[cursor:start])
        collapsed.extend(placeholder)
        cursor = end + 1
    collapsed.extend(tokens[cursor:])
    return hf_tokenizer.decode(collapsed, skip_special_tokens=False)


def build_multimodal_train_inputs_from_traj(
    traj: dict[str, Any],
    processor: Any,
    *,
    traj_path: str | os.PathLike[str],
    osworld_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run HF processor on collapsed prompt + images; output must align with traj token ids."""
    from slime.utils.processing_utils import build_processor_kwargs

    tokens: list[int] = list(traj["tokens"])
    spans = traj.get("image_position_spans", [])
    num_images = len(traj.get("metadata", {}).get("images", []))
    if num_images > 0:
        if not spans:
            raise ValueError("traj has images but no image_position_spans")
        text = collapse_vision_tokens_for_processor(tokens, spans, processor)
    else:
        text = traj.get("tokens_decoded") or processor.tokenizer.decode(tokens, skip_special_tokens=False)

    image_paths = [
        resolve_traj_asset_path(img["path"], traj_path=traj_path, osworld_root=osworld_root)
        for img in traj.get("metadata", {}).get("images", [])
    ]
    multimodal_inputs = _build_multimodal_inputs_from_image_paths(image_paths)
    processor_output = processor(text=text, **build_processor_kwargs(multimodal_inputs))
    proc_ids = processor_output["input_ids"][0]
    if hasattr(proc_ids, "tolist"):
        proc_ids = proc_ids.tolist()
    else:
        proc_ids = list(proc_ids)
    if len(proc_ids) != len(traj["tokens"]):
        raise ValueError(
            f"Processor token length {len(proc_ids)} != traj tokens {len(traj['tokens'])}; "
            "training inputs would not align with rollout tokens."
        )
    return {
        key: value
        for key, value in processor_output.items()
        if key not in _PROCESSOR_PROMPT_KEYS
    } or {}


def validate_traj_for_sample(traj: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty means OK)."""
    errors: list[str] = []
    required = ("tokens", "response_length", "loss_mask", "reward")
    for key in required:
        if key not in traj:
            errors.append(f"missing key: {key}")

    tokens = traj.get("tokens", [])
    response_length = traj.get("response_length", 0)
    loss_mask = traj.get("loss_mask", [])

    if response_length > len(tokens):
        errors.append(f"response_length {response_length} > len(tokens) {len(tokens)}")
    if len(loss_mask) != response_length:
        errors.append(f"len(loss_mask)={len(loss_mask)} != response_length={response_length}")

    spans = traj.get("image_position_spans", [])
    num_images = traj.get("metadata", {}).get("num_images", len(traj.get("metadata", {}).get("images", [])))
    if num_images > 0:
        if len(spans) < num_images:
            errors.append(f"vision spans {len(spans)} < num_images {num_images}")
        for span in spans:
            if span[1] - span[0] + 1 <= 3:
                errors.append(f"vision span {span} not expanded (placeholder-only)")

    return errors


def validate_sample_against_traj(sample: Sample, traj: dict[str, Any]) -> list[str]:
    errors = validate_traj_for_sample(traj)
    if sample.tokens != traj["tokens"]:
        errors.append(f"sample.tokens length {len(sample.tokens)} != traj {len(traj['tokens'])}")
    if sample.response_length != traj["response_length"]:
        errors.append("sample.response_length mismatch")
    if sample.loss_mask != traj["loss_mask"]:
        errors.append("sample.loss_mask mismatch")
    return errors


def traj_to_sample(
    traj_path: str | os.PathLike[str],
    *,
    osworld_root: str | os.PathLike[str] | None = None,
    processor: Any | None = None,
    tokenizer: Any | None = None,
    build_multimodal_train_inputs: bool = True,
) -> Sample:
    """Convert one OSWorld traj JSON file to a completed slime Sample for GRPO training."""
    traj_path = os.path.abspath(traj_path)
    traj = load_traj_json(traj_path)
    errors = validate_traj_for_sample(traj)
    if errors:
        raise ValueError(f"Invalid traj {traj_path}: {'; '.join(errors)}")

    osworld_root = osworld_root or os.environ.get("OSWORLD_ROOT") or infer_osworld_root(traj_path)

    tokens: list[int] = list(traj["tokens"])
    response_length: int = int(traj["response_length"])
    loss_mask: list[int] = list(traj["loss_mask"])

    response = ""
    if tokenizer is not None and response_length > 0:
        response = tokenizer.decode(tokens[-response_length:], skip_special_tokens=False)

    multimodal_inputs = None
    image_paths: list[str] = []
    for img in traj.get("metadata", {}).get("images", []):
        image_paths.append(
            resolve_traj_asset_path(img["path"], traj_path=traj_path, osworld_root=osworld_root)
        )
    if image_paths:
        multimodal_inputs = _build_multimodal_inputs_from_image_paths(image_paths)

    multimodal_train_inputs = None
    if build_multimodal_train_inputs:
        if processor is None:
            raise ValueError("processor is required when build_multimodal_train_inputs=True")
        multimodal_train_inputs = build_multimodal_train_inputs_from_traj(
            traj,
            processor,
            traj_path=traj_path,
            osworld_root=osworld_root,
        )

    metadata = dict(traj.get("metadata", {}))
    metadata["traj_path"] = traj_path
    if osworld_root:
        metadata["osworld_root"] = str(osworld_root)

    return Sample(
        prompt=traj.get("tokens_decoded", ""),
        tokens=tokens,
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=multimodal_train_inputs,
        response=response,
        response_length=response_length,
        reward=traj.get("reward"),
        loss_mask=loss_mask,
        status=Sample.Status.COMPLETED,
        metadata=metadata,
    )


def load_traj_paths_as_samples(
    traj_paths: list[str | os.PathLike[str]],
    *,
    osworld_root: str | os.PathLike[str] | None = None,
    processor: Any | None = None,
    tokenizer: Any | None = None,
    build_multimodal_train_inputs: bool = True,
) -> list[Sample]:
    return [
        traj_to_sample(
            path,
            osworld_root=osworld_root,
            processor=processor,
            tokenizer=tokenizer,
            build_multimodal_train_inputs=build_multimodal_train_inputs,
        )
        for path in traj_paths
    ]


def samples_to_train_data_dict(samples: list[Sample]) -> dict[str, Any]:
    """Mirror slime.ray.rollout.RolloutManager._convert_samples_to_train_data for smoke tests."""
    if not samples:
        return {"tokens": [], "response_lengths": [], "rewards": [], "loss_masks": []}

    for sample in samples:
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        assert len(sample.loss_mask) == sample.response_length

    data: dict[str, Any] = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": [sample.reward for sample in samples],
        "loss_masks": [sample.loss_mask for sample in samples],
    }
    if any(sample.multimodal_train_inputs is not None for sample in samples):
        data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]
    data["total_lengths"] = [len(t) for t in data["tokens"]]
    return data
