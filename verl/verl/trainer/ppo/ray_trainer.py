# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import shutil
import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig, DATA2KEYS


WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Interaction = 7


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def compute_data_metrics_(batch, use_critic=True):
    metrics = {}
    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        turns = np.maximum(np.array(batch.meta_info['turns_stats'], dtype=np.float32), 1.0)
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.float32) / turns).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())
    for itm in ['early_cut', 'repeats', 'stalling', 'must_stop']:
        if itm in batch.meta_info:
            metrics[f'env/{itm}'] = float(np.array(batch.meta_info[itm]).mean())
    for itm in ['tau2_invalid_actions', 'tau2_tool_errors', 'tau2_no_state_change_writes']:
        if itm in batch.meta_info:
            metrics[f'env/{itm}'] = float(np.array(batch.meta_info[itm], dtype=np.float32).mean())
    for itm in ['tau2_hard_truncations', 'tau2_soft_truncations']:
        if itm in batch.meta_info:
            values = np.array(batch.meta_info[itm], dtype=np.float32)
            metrics[f'env/{itm}'] = float(values.sum())
            metrics[f'env/{itm}_ratio'] = float(values.mean())
    for itm in ['tau2_positive_steps', 'tau2_neutral_steps', 'tau2_negative_steps']:
        if itm in batch.meta_info:
            metrics[f'env/{itm}'] = float(np.array(batch.meta_info[itm], dtype=np.float32).mean())
    for itm in ['tau2_arew_positive_steps', 'tau2_arew_neutral_steps', 'tau2_arew_negative_steps']:
        if itm in batch.meta_info:
            metrics[f'env/{itm}'] = float(np.array(batch.meta_info[itm], dtype=np.float32).mean())
    if 'tau2_arew_has_pos_neg' in batch.meta_info:
        metrics['env/tau2_arew_pos_neg_trajectory_ratio'] = float(
            np.array(batch.meta_info['tau2_arew_has_pos_neg'], dtype=np.float32).mean()
        )
    for itm in [
        'tau2_message_turns',
        'tau2_tool_turns',
        'tau2_read_tool_turns',
        'tau2_write_tool_turns',
        'tau2_user_tool_hops',
        'tau2_bootstrap_user_tool_hops',
        'tau2_user_disconnects',
        'tau2_bootstrap_user_disconnects',
        'tau2_disconnect_repeat_neutral_steps',
        'tau2_assistant_tool_errors',
        'tau2_user_tool_errors',
        'tau2_state_changed_steps',
        'tau2_action_progress_steps',
        'tau2_new_raw_information_steps',
        'tau2_new_normalized_information_steps',
        'tau2_new_family_steps',
        'tau2_max_no_progress_streak',
        'tau2_t3_truncation_turn',
        'tau2_user_stopped',
    ]:
        if itm in batch.meta_info:
            metrics[f'env/{itm}'] = float(np.array(batch.meta_info[itm], dtype=np.float32).mean())
    if 'turns_stats' in batch.meta_info:
        turns_stats = np.array(batch.meta_info['turns_stats'], dtype=np.float32)
        if 'tau2_arew_positive_steps' in batch.meta_info:
            values = np.array(batch.meta_info['tau2_arew_positive_steps'], dtype=np.float32)
            metrics['env/tau2_arew_positive_step_ratio'] = float((values / np.maximum(turns_stats, 1.0)).mean())
        if 'tau2_arew_neutral_steps' in batch.meta_info:
            values = np.array(batch.meta_info['tau2_arew_neutral_steps'], dtype=np.float32)
            metrics['env/tau2_arew_neutral_step_ratio'] = float((values / np.maximum(turns_stats, 1.0)).mean())
        if 'tau2_arew_negative_steps' in batch.meta_info:
            values = np.array(batch.meta_info['tau2_arew_negative_steps'], dtype=np.float32)
            metrics['env/tau2_arew_negative_step_ratio'] = float((values / np.maximum(turns_stats, 1.0)).mean())
    if "arew_spos" in batch.batch:
        metrics["arew/spos_mean"] = float(batch.batch["arew_spos"].float().mean().item())
    if "arew_sneg" in batch.batch:
        metrics["arew/sneg_mean"] = float(batch.batch["arew_sneg"].float().mean().item())
    if "arew_pos_steps" in batch.batch:
        metrics["arew/pos_step_count_mean"] = float(batch.batch["arew_pos_steps"].float().mean().item())
    if "arew_neg_steps" in batch.batch:
        metrics["arew/neg_step_count_mean"] = float(batch.batch["arew_neg_steps"].float().mean().item())
    if "arew_raw_pos_steps" in batch.batch:
        metrics["arew/raw_pos_step_count_mean"] = float(batch.batch["arew_raw_pos_steps"].float().mean().item())
    if "arew_raw_neg_steps" in batch.batch:
        metrics["arew/raw_neg_step_count_mean"] = float(batch.batch["arew_raw_neg_steps"].float().mean().item())
    if "arew_eff_pos_steps" in batch.batch:
        metrics["arew/effective_pos_step_count_mean"] = float(batch.batch["arew_eff_pos_steps"].float().mean().item())
    if "arew_eff_neg_steps" in batch.batch:
        metrics["arew/effective_neg_step_count_mean"] = float(batch.batch["arew_eff_neg_steps"].float().mean().item())
    if "arew_raw_pos_steps" in batch.batch and "arew_raw_neg_steps" in batch.batch:
        raw_pos_steps = batch.batch["arew_raw_pos_steps"]
        raw_neg_steps = batch.batch["arew_raw_neg_steps"]
        metrics["arew/raw_both_sided_ratio"] = float(((raw_pos_steps > 0) & (raw_neg_steps > 0)).float().mean().item())
        metrics["arew/raw_pos_only_ratio"] = float(((raw_pos_steps > 0) & (raw_neg_steps == 0)).float().mean().item())
        metrics["arew/raw_neg_only_ratio"] = float(((raw_pos_steps == 0) & (raw_neg_steps > 0)).float().mean().item())
    if "arew_eff_pos_steps" in batch.batch and "arew_eff_neg_steps" in batch.batch:
        eff_pos_steps = batch.batch["arew_eff_pos_steps"]
        eff_neg_steps = batch.batch["arew_eff_neg_steps"]
        metrics["arew/effective_active_ratio"] = float(((eff_pos_steps > 0) | (eff_neg_steps > 0)).float().mean().item())
        metrics["arew/effective_both_sided_ratio"] = float(
            ((eff_pos_steps > 0) & (eff_neg_steps > 0)).float().mean().item()
        )
        metrics["arew/effective_pos_only_ratio"] = float(
            ((eff_pos_steps > 0) & (eff_neg_steps == 0)).float().mean().item()
        )
        metrics["arew/effective_neg_only_ratio"] = float(
            ((eff_pos_steps == 0) & (eff_neg_steps > 0)).float().mean().item()
        )
    return metrics


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def get_arew_modifier(
    bonus_labels: torch.Tensor,
    advantages: torch.Tensor,
    config: Optional[AlgoConfig],
    step_ids: Optional[torch.Tensor] = None,
    complementary_neutral_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Map token-level {-1,0,+1} labels to a zero-sum advantage bonus."""
    mode = getattr(config, "arew_bonus_mode", "minority_fixed")
    scale = float(getattr(config, "arew_bonus_scale", 0.1))
    negative_step_mode = getattr(config, "arew_negative_step_mode", "full")

    is_pos = bonus_labels.gt(0)
    is_neg = bonus_labels.lt(0)
    s_pos = is_pos.sum(dim=1)
    s_neg = is_neg.sum(dim=1)
    valid = (s_pos > 0) & (s_neg > 0)
    pos_step_counts = torch.zeros(bonus_labels.size(0), device=bonus_labels.device, dtype=torch.int32)
    neg_step_counts = torch.zeros(bonus_labels.size(0), device=bonus_labels.device, dtype=torch.int32)
    if step_ids is not None:
        step_ids = step_ids.to(device=bonus_labels.device)
        for row_idx in range(bonus_labels.size(0)):
            row_step_ids = step_ids[row_idx]
            pos_step_ids = torch.unique(row_step_ids[bonus_labels[row_idx] > 0])
            neg_step_ids = torch.unique(row_step_ids[bonus_labels[row_idx] < 0])
            pos_step_counts[row_idx] = int((pos_step_ids > 0).sum().item())
            neg_step_counts[row_idx] = int((neg_step_ids > 0).sum().item())
    raw_pos_step_counts = pos_step_counts.clone()
    raw_neg_step_counts = neg_step_counts.clone()

    s_pos_f = s_pos.to(advantages.dtype)
    s_neg_f = s_neg.to(advantages.dtype)

    if mode == "fixed_pos":
        pos = torch.full((bonus_labels.size(0),), scale, device=bonus_labels.device, dtype=advantages.dtype)
        neg = (s_pos_f / (s_neg_f + eps)) * pos
    elif mode == "fixed_neg":
        neg = torch.full((bonus_labels.size(0),), scale, device=bonus_labels.device, dtype=advantages.dtype)
        pos = (s_neg_f / (s_pos_f + eps)) * neg
    elif mode == "abs_sum":
        pos = torch.where(
            s_pos > 0,
            torch.full_like(s_pos_f, scale) / (2.0 * (s_pos_f + eps)),
            torch.zeros_like(s_pos_f),
        )
        neg = torch.where(
            s_neg > 0,
            torch.full_like(s_neg_f, scale) / (2.0 * (s_neg_f + eps)),
            torch.zeros_like(s_neg_f),
        )
    elif mode == "minority_fixed":
        fix_pos_mask = s_pos <= s_neg
        pos_fixed = torch.full((bonus_labels.size(0),), scale, device=bonus_labels.device, dtype=advantages.dtype)
        neg_fixed = torch.full((bonus_labels.size(0),), scale, device=bonus_labels.device, dtype=advantages.dtype)
        neg_from_pos = (s_pos_f / (s_neg_f + eps)) * pos_fixed
        pos_from_neg = (s_neg_f / (s_pos_f + eps)) * neg_fixed
        pos = torch.where(fix_pos_mask, pos_fixed, pos_from_neg)
        neg = torch.where(fix_pos_mask, neg_from_pos, neg_fixed)
    elif mode == "step_abs_sum":
        if step_ids is None:
            raise ValueError("AREW mode=step_abs_sum requires responses_arew_step_ids in batch.")
        if complementary_neutral_mask is not None:
            complementary_neutral_mask = complementary_neutral_mask.to(device=bonus_labels.device).bool()
        if negative_step_mode not in {"full", "prefix", "suffix"}:
            raise ValueError(
                f"Unknown AREW negative-step mode={negative_step_mode}. Choose from full|prefix|suffix"
            )
        shaped = torch.zeros_like(bonus_labels, dtype=advantages.dtype)
        selected_s_pos = torch.zeros_like(s_pos)
        selected_s_neg = torch.zeros_like(s_neg)
        effective_pos_step_counts = torch.zeros_like(pos_step_counts)
        effective_neg_step_counts = torch.zeros_like(neg_step_counts)
        for row_idx in range(bonus_labels.size(0)):
            row_labels = bonus_labels[row_idx]
            row_step_ids = step_ids[row_idx]
            pos_step_ids = torch.unique(row_step_ids[row_labels > 0])
            neg_step_ids = torch.unique(row_step_ids[row_labels < 0])
            pos_step_ids = pos_step_ids[pos_step_ids > 0]
            neg_step_ids = neg_step_ids[neg_step_ids > 0]

            positive_step_ids = pos_step_ids
            negative_step_ids = neg_step_ids
            synthetic_positive_step_ids = pos_step_ids.new_empty((0,), dtype=pos_step_ids.dtype)
            synthetic_negative_step_ids = neg_step_ids.new_empty((0,), dtype=neg_step_ids.dtype)
            positive_on_neutral = False
            negative_on_neutral = False

            positive_step_id_values = positive_step_ids.tolist()
            negative_step_id_values = negative_step_ids.tolist()
            final_step_id = None
            positive_row_step_ids = row_step_ids[row_step_ids > 0]
            if positive_row_step_ids.numel() > 0:
                final_step_id = int(positive_row_step_ids.max().item())
            neutral_mask = row_labels.eq(0) & row_step_ids.gt(0)
            if final_step_id is not None:
                neutral_mask = neutral_mask & row_step_ids.ne(final_step_id)
            neutral_step_ids = torch.unique(row_step_ids[neutral_mask])
            neutral_step_ids = neutral_step_ids[neutral_step_ids > 0]
            complementary_neutral_step_ids = neutral_step_ids
            if complementary_neutral_mask is not None:
                complementary_neutral_step_ids = torch.unique(
                    row_step_ids[neutral_mask & complementary_neutral_mask[row_idx]]
                )
                complementary_neutral_step_ids = complementary_neutral_step_ids[complementary_neutral_step_ids > 0]

            if (
                positive_step_ids.numel() == 0
                and negative_step_ids.numel() > 0
                and complementary_neutral_step_ids.numel() > 0
            ):
                synthetic_positive_step_ids = complementary_neutral_step_ids
                positive_on_neutral = True
            elif negative_step_ids.numel() == 0 and positive_step_ids.numel() > 0 and neutral_step_ids.numel() > 0:
                synthetic_negative_step_ids = neutral_step_ids
                negative_on_neutral = True

            effective_positive_step_ids = positive_step_ids if positive_step_ids.numel() > 0 else synthetic_positive_step_ids
            effective_negative_step_ids = negative_step_ids if negative_step_ids.numel() > 0 else synthetic_negative_step_ids
            if effective_positive_step_ids.numel() == 0 or effective_negative_step_ids.numel() == 0:
                continue
            if negative_step_mode != "full" and effective_negative_step_ids.numel() > effective_positive_step_ids.numel():
                keep_n = int(effective_positive_step_ids.numel())
                if negative_step_mode == "prefix":
                    effective_negative_step_ids = effective_negative_step_ids[:keep_n]
                else:
                    effective_negative_step_ids = effective_negative_step_ids[-keep_n:]

            effective_pos_step_counts[row_idx] = int(effective_positive_step_ids.numel())
            effective_neg_step_counts[row_idx] = int(effective_negative_step_ids.numel())

            pos_step_budget = scale / (2.0 * float(effective_positive_step_ids.numel()))
            neg_step_budget = scale / (2.0 * float(effective_negative_step_ids.numel()))
            for step_id in effective_positive_step_ids.tolist():
                if positive_on_neutral and step_id not in positive_step_id_values:
                    step_mask = row_step_ids.eq(step_id) & row_labels.eq(0)
                else:
                    step_mask = row_step_ids.eq(step_id) & row_labels.gt(0)
                step_len = int(step_mask.sum().item())
                if step_len > 0:
                    shaped[row_idx, step_mask] = pos_step_budget / step_len
                    selected_s_pos[row_idx] += step_len
            for step_id in effective_negative_step_ids.tolist():
                if negative_on_neutral and step_id not in negative_step_id_values:
                    step_mask = row_step_ids.eq(step_id) & row_labels.eq(0)
                else:
                    step_mask = row_step_ids.eq(step_id) & row_labels.lt(0)
                step_len = int(step_mask.sum().item())
                if step_len > 0:
                    shaped[row_idx, step_mask] = -neg_step_budget / step_len
                    selected_s_neg[row_idx] += step_len
            pos_step_counts[row_idx] = int(effective_positive_step_ids.numel())
            neg_step_counts[row_idx] = int(effective_negative_step_ids.numel())
        valid = (selected_s_pos > 0) & (selected_s_neg > 0)
        shaped = torch.where(valid.unsqueeze(1), shaped, torch.zeros_like(shaped))
        return (
            shaped.to(advantages.dtype),
            selected_s_pos,
            selected_s_neg,
            pos_step_counts,
            neg_step_counts,
            raw_pos_step_counts,
            raw_neg_step_counts,
            effective_pos_step_counts,
            effective_neg_step_counts,
        )
    else:
        raise ValueError(
            f"Unknown AREW mode={mode}. Choose from fixed_pos|fixed_neg|abs_sum|minority_fixed|step_abs_sum"
        )

    pos_bt = pos.unsqueeze(1)
    neg_bt = neg.unsqueeze(1)
    shaped = torch.zeros_like(bonus_labels, dtype=advantages.dtype)
    shaped = torch.where(is_pos, pos_bt.expand_as(shaped), shaped)
    shaped = torch.where(is_neg, -neg_bt.expand_as(shaped), shaped)
    shaped = torch.where(valid.unsqueeze(1), shaped, torch.zeros_like(shaped))

    return (
        shaped.to(advantages.dtype),
        s_pos,
        s_neg,
        pos_step_counts,
        neg_step_counts,
        raw_pos_step_counts,
        raw_neg_step_counts,
        pos_step_counts,
        neg_step_counts,
    )


# def remap_arew_labels(
#     bonus_labels: torch.Tensor,
#     response_mask: torch.Tensor,
#     config: Optional[AlgoConfig],
# ) -> torch.Tensor:
#     if config is None:
#         return bonus_labels
#     if getattr(config, "arew_negative_on_neutral", False):
#         valid_response_tokens = response_mask.to(dtype=torch.bool, device=bonus_labels.device)
#         should_flip = (bonus_labels == 0) & valid_response_tokens
#         return torch.where(should_flip, torch.full_like(bonus_labels, -1), bonus_labels)
#     return bonus_labels


def is_arew_active(config: Optional[AlgoConfig], global_step: int) -> bool:
    if config is None or not config.get("use_arew_bonus", False):
        return False
    start_step = int(getattr(config, "arew_start_step", 0))
    end_step = int(getattr(config, "arew_end_step", -1))
    if global_step < start_step:
        return False
    if end_step >= 0 and global_step >= end_step:
        return False
    return True


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    save_adv_name: str = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        current_global_step = int(data.meta_info.get("global_steps", 0))
        if is_arew_active(config, current_global_step) and "responses_arew_labels" in data.batch:
            arew_labels = data.batch["responses_arew_labels"].to(device=advantages.device, dtype=advantages.dtype)
            # arew_labels = remap_arew_labels(
            #     arew_labels,
            #     data.batch["response_mask"].to(device=advantages.device),
            #     config,
            # )
            if arew_labels.shape != advantages.shape:
                raise ValueError(
                    f"responses_arew_labels shape {arew_labels.shape} does not match advantages shape {advantages.shape}"
                )
            arew_step_ids = data.batch.get("responses_arew_step_ids", None)
            if arew_step_ids is not None:
                arew_step_ids = arew_step_ids.to(device=advantages.device)
            complementary_neutral_mask = data.batch.get("responses_arew_complementary_mask", None)
            if complementary_neutral_mask is not None:
                complementary_neutral_mask = complementary_neutral_mask.to(device=advantages.device)
            (
                modifier,
                s_pos,
                s_neg,
                pos_steps,
                neg_steps,
                raw_pos_steps,
                raw_neg_steps,
                eff_pos_steps,
                eff_neg_steps,
            ) = get_arew_modifier(
                arew_labels,
                advantages,
                config,
                step_ids=arew_step_ids,
                complementary_neutral_mask=complementary_neutral_mask,
            )
            modifier = modifier * data.batch["response_mask"].to(modifier.dtype)
            data.batch["arew_spos"] = s_pos
            data.batch["arew_sneg"] = s_neg
            data.batch["arew_pos_steps"] = pos_steps
            data.batch["arew_neg_steps"] = neg_steps
            data.batch["arew_raw_pos_steps"] = raw_pos_steps
            data.batch["arew_raw_neg_steps"] = raw_neg_steps
            data.batch["arew_eff_pos_steps"] = eff_pos_steps
            data.batch["arew_eff_neg_steps"] = eff_neg_steps
            data.batch["advantages"] = advantages + modifier
        else:
            data.batch["advantages"] = advantages

        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        current_global_step = int(data.meta_info.get("global_steps", 0))
        if is_arew_active(config, current_global_step) and "responses_arew_labels" in data.batch:
            arew_labels = data.batch["responses_arew_labels"].to(device=advantages.device, dtype=advantages.dtype)
            # arew_labels = remap_arew_labels(
            #     arew_labels,
            #     data.batch["response_mask"].to(device=advantages.device),
            #     config,
            # )
            if arew_labels.shape != advantages.shape:
                raise ValueError(
                    f"responses_arew_labels shape {arew_labels.shape} does not match advantages shape {advantages.shape}"
                )
            arew_step_ids = data.batch.get("responses_arew_step_ids", None)
            if arew_step_ids is not None:
                arew_step_ids = arew_step_ids.to(device=advantages.device)
            complementary_neutral_mask = data.batch.get("responses_arew_complementary_mask", None)
            if complementary_neutral_mask is not None:
                complementary_neutral_mask = complementary_neutral_mask.to(device=advantages.device)
            (
                modifier,
                s_pos,
                s_neg,
                pos_steps,
                neg_steps,
                raw_pos_steps,
                raw_neg_steps,
                eff_pos_steps,
                eff_neg_steps,
            ) = get_arew_modifier(
                arew_labels,
                advantages,
                config,
                step_ids=arew_step_ids,
                complementary_neutral_mask=complementary_neutral_mask,
            )
            modifier = modifier * data.batch["response_mask"].to(modifier.dtype)
            data.batch["arew_spos"] = s_pos
            data.batch["arew_sneg"] = s_neg
            data.batch["arew_pos_steps"] = pos_steps
            data.batch["arew_neg_steps"] = neg_steps
            data.batch["arew_raw_pos_steps"] = raw_pos_steps
            data.batch["arew_raw_neg_steps"] = raw_neg_steps
            data.batch["arew_eff_pos_steps"] = eff_pos_steps
            data.batch["arew_eff_neg_steps"] = eff_neg_steps
            data.batch["advantages"] = advantages + modifier
        else:
            data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None,
                 collate_fn=None,
                 train_sampler: Optional[Sampler] = None,
                 device_name=None,):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        if "console" in self.config.trainer.logger:
            print(f"Validation trajectories at step {self.global_steps}:")
            for idx, (input_text, output_text, score) in enumerate(samples, start=1):
                print(f"[val sample {idx}] id={input_text} score={float(score):.4f}")
                print(output_text)
                print("-" * 80)

        # Log to each configured logger
        non_console_loggers = [logger for logger in self.config.trainer.logger if logger != "console"]
        if non_console_loggers:
            self.validation_generations_logger.log(non_console_loggers, samples, self.global_steps)

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []
        counts_lst = []
        val_log_inputs = []
        val_log_outputs = []
        val_log_scores = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node,
            dataset = self.config.data_name,
            early_cut = self.config.early_cut,
            trunc_strength = self.config.trunc_strength,
            hard_tolerate_num = OmegaConf.select(self.config, "hard_tolerate_num", default=2),
            strict_progress_label = OmegaConf.select(self.config, "tau2_strict_progress_label", default=False),
            completion_bonus = OmegaConf.select(self.config, "tau2_completion_bonus", default=0.0),
            tau2_t3_progress_mode = OmegaConf.select(self.config, "tau2_t3_progress_mode", default="legacy"),
            tau2_t3_min_turn = OmegaConf.select(self.config, "tau2_t3_min_turn", default=6),
            tau2_arew_label_mode = OmegaConf.select(self.config, "tau2_arew_label_mode", default="clean"),
            tau2_arew_min_turn = OmegaConf.select(self.config, "tau2_arew_min_turn", default=8),
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )
        for batch_dict in self.val_dataloader:

            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
            
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                            non_tensor_batch_keys=DATA2KEYS[self.config.data_name])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
            final_gen_batch_output = generation_manager.run_llm_loop(
                gen_batch=test_gen_batch,
                initial_input_ids=first_input_ids,
                validate=True
            )
            counts_lst.append(final_gen_batch_output.meta_info['turns_stats'])
            test_batch = test_batch.union(final_gen_batch_output)
            
            for key in test_batch.batch.keys():
                test_batch.batch[key] = test_batch.batch[key].long()
            
            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            if self.config.trainer.log_val_generations > 0:
                batch_scores = reward_tensor.sum(-1).float().cpu().tolist()
                task_ids = test_batch.non_tensor_batch.get("tau2_task_id", None)
                trajectories = test_batch.non_tensor_batch.get("tau2_trajectory", None)
                if task_ids is not None and trajectories is not None:
                    val_log_inputs.extend(task_ids.tolist())
                    val_log_outputs.extend(trajectories.tolist())
                    val_log_scores.extend(batch_scores)

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).float().cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        counts_lst = np.concatenate(counts_lst, axis=0)
        assert len(counts_lst) == len(reward_tensor)
        # evaluate test_score based on data source
        metric_dict = {
            'val/success_rate': torch.mean(reward_tensor).item(),
            'val/counts_taken': float(np.mean(counts_lst)),
        }
        if val_log_outputs:
            self._maybe_log_val_generations(val_log_inputs, val_log_outputs, val_log_scores)
        print(f"Validation metrics at step {self.global_steps}: {metric_dict}")
        print("Validation Done!")
        # data_source_reward = {}
        # for i in range(reward_tensor.shape[0]):
        #     data_source = data_sources[i]
        #     if data_source not in data_source_reward:
        #         data_source_reward[data_source] = []
        #     data_source_reward[data_source].append(reward_tensor[i].item())

        # metric_dict = {}
        # for data_source, rewards in data_source_reward.items():
        #     metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint_best(self):
        # Remove old best models
        best_models_dir = os.path.join(self.config.trainer.default_local_dir, 'actor')
        if os.path.exists(best_models_dir):
            for item in os.listdir(best_models_dir):
                if item.startswith('best_step_'):
                    item_path = os.path.join(best_models_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)

        # Save new best model
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'best_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node,
            dataset = self.config.data_name,
            early_cut = self.config.early_cut,
            trunc_strength = self.config.trunc_strength,
            hard_tolerate_num = OmegaConf.select(self.config, "hard_tolerate_num", default=2),
            strict_progress_label = OmegaConf.select(self.config, "tau2_strict_progress_label", default=False),
            completion_bonus = OmegaConf.select(self.config, "tau2_completion_bonus", default=0.0),
            tau2_t3_progress_mode = OmegaConf.select(self.config, "tau2_t3_progress_mode", default="legacy"),
            tau2_t3_min_turn = OmegaConf.select(self.config, "tau2_t3_min_turn", default=6),
            tau2_arew_label_mode = OmegaConf.select(self.config, "tau2_arew_label_mode", default="clean"),
            tau2_arew_min_turn = OmegaConf.select(self.config, "tau2_arew_min_turn", default=8),
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config
        )
        self.best_reward = float('-inf')
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch.batch['input_ids'] = batch.batch['input_ids'].long()
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                                      non_tensor_batch_keys=DATA2KEYS[self.config.data_name])

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                            global_step=self.global_steps
                        )
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    else:
                        response_length = batch.batch['responses'].shape[-1]
                        # response_mask = batch.batch['attention_mask'][:, -response_length:]
                        batch.batch['response_mask'] = batch.batch['response_mask'][:, -response_length:]                       
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        batch.batch['input_ids'] = batch.batch['input_ids'].long()
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        # TODO: combine different dimensions of rewards here ??
                        reward_tensor, final_rewards = self.reward_fn(batch)
                        # refine_reward_tensor = self.reward_fn.get_refine_subem(batch)
                        # batch.batch.update(self.reward_fn.get_logging_scores(batch, self.global_steps))
                        batch.batch['token_level_scores'] = reward_tensor
                        batch.non_tensor_batch['final_rewards'] = final_rewards
                        # batch.batch['token_level_refine_scores'] = refine_reward_tensor

                        # if self.config.actor_rollout_ref.actor.refine_lambda > 0:
                        #     reward_tensor += self.config.actor_rollout_ref.actor.refine_lambda * refine_reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            save_adv_name=self.config.trainer.experiment_name
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and self.global_steps > 2 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics_(batch=batch))
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                if metrics['critic/score/mean'] > self.best_reward:

                    if self.global_steps % self.config.trainer.test_freq != 0:
                        val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                    if self.global_steps > 2:
                        self._save_checkpoint_best()
                    self.best_reward = max(self.best_reward, metrics['critic/score/mean'])

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
