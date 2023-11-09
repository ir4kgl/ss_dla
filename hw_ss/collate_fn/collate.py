import torch
import logging
from typing import List

from torch.nn import ConstantPad2d

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    result_batch["audio_len"] = list(x["audio_len"] for x in dataset_items)
    result_batch["ref_len"] = list(x["ref_len"] for x in dataset_items)
    result_batch["target_len"] = list(x["target_len"] for x in dataset_items)
    result_batch["speaker_id"] = torch.Tensor(
        list(x["target_id"] for x in dataset_items))

    batch_audio_len = min(result_batch["audio_len"])
    batch_target_len = min(result_batch["target_len"])
    batch_ref_len = min(result_batch["ref_len"])

    batch_audio_len = min(batch_audio_len, batch_target_len)
    batch_target_len = min(batch_audio_len, batch_target_len)

    result_batch["audio"] = torch.cat(
        tuple(x["audio"][:, :batch_audio_len] for x in dataset_items)
    ).unsqueeze(1)
    result_batch["target"] = torch.cat(
        tuple(x["target"][:, :batch_target_len] for x in dataset_items)
    ).unsqueeze(1)
    result_batch["ref"] = torch.cat(
        tuple(x["ref"][:, :batch_ref_len] for x in dataset_items)
    ).unsqueeze(1)

    return result_batch


def collate_fn_test(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    result_batch["audio_len"] = list(x["audio_len"] for x in dataset_items)
    result_batch["ref_len"] = list(x["ref_len"] for x in dataset_items)
    result_batch["target_len"] = list(x["target_len"] for x in dataset_items)
    result_batch["speaker_id"] = torch.Tensor(
        list(x["target_id"] for x in dataset_items))
    batch_audio_len = max(result_batch["audio_len"])
    batch_target_len = max(result_batch["target_len"])
    batch_ref_len = max(result_batch["ref_len"])

    batch_audio_len = max(batch_audio_len, batch_target_len)
    batch_target_len = max(batch_audio_len, batch_target_len)

    result_batch["audio"] = torch.cat(
        tuple(ConstantPad2d((0, batch_audio_len - x["audio_len"], 0, 0), 0)(x["audio"]) for x in dataset_items)).unsqueeze(1)

    result_batch["target"] = torch.cat(
        tuple(ConstantPad2d((0, batch_target_len - x["target_len"], 0, 0), 0)(x["target"]) for x in dataset_items)).unsqueeze(1)

    result_batch["ref"] = torch.cat(
        tuple(ConstantPad2d((0, batch_ref_len - x["ref_len"], 0, 0), 0)(x["ref"]) for x in dataset_items)).unsqueeze(1)

    return result_batch
