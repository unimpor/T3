import torch
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from .utils import CircuitDecodingSpace, MovieRecSpace
from search_r1.tau2_adapter.space import Tau2SoloSpace
from search_r1.tau2_adapter.standard_space import Tau2StandardSpace
from verl import DataProto

DATA2KEYS = {
    "CircuitDecoding": ['controller'],
    "MovieRec": ["controller"],
    "Tau2Bench": ["controller"],
}

# llama template:

# start_user = "<|start_header_id|>user<|end_header_id|>\n\n"
# start_assistant = "<|start_header_id|>assistant<|end_header_id|>\n\n"
# end_signal = "<|eot_id|>"

# qwen template:

start_user = "\n<|im_start|>user\n"
start_assistant = "\n<|im_start|>assistant\n"
start_tool = "\n<|im_start|>tool\n"
end_signal = "<|im_end|>"

_ROLE_PREFIX = {
    "user": start_user,
    "assistant": start_assistant,
    "tool": start_tool,
}
_TAU2_STANDARD_USER_SIM_PARALLELISM = max(1, int(os.getenv("TAU2_STANDARD_USER_SIM_PARALLELISM", "16")))


def _normalize_chat_content(content: Any) -> str:
    return "" if content is None else str(content).strip()


def _render_chat_observation(role: str, content: Any) -> str:
    if role not in _ROLE_PREFIX:
        raise ValueError(f"Unsupported chat role: {role}")
    return f"{_ROLE_PREFIX[role]}{_normalize_chat_content(content)}\n{end_signal}{start_assistant}"


def _render_standard_bootstrap(turns: List[Tuple[str, str]]) -> str:
    if not turns:
        return ""

    first_role, first_content = turns[0]
    if first_role != "assistant":
        raise ValueError(f"Standard bootstrap must start with assistant, got role={first_role}")

    chunks = [f"{_normalize_chat_content(first_content)}\n{end_signal}"]
    for role, content in turns[1:]:
        if role not in _ROLE_PREFIX:
            raise ValueError(f"Unsupported bootstrap role: {role}")
        chunks.append(f"{_ROLE_PREFIX[role]}{_normalize_chat_content(content)}\n{end_signal}")
    chunks.append(start_assistant)
    return "".join(chunks)


def _truncate_response_to_action(response: str) -> str:
    closing_tags = ["</answer>", "</interact>", "</message>"]
    matches = [(response.find(tag), tag) for tag in closing_tags if response.find(tag) != -1]
    if not matches:
        return response
    idx, tag = min(matches, key=lambda item: item[0])
    return response[: idx + len(tag)] + end_signal

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    dataset: str = ''
    early_cut: bool = False
    trunc_strength: int = 0
    hard_tolerate_num: int = 2
    strict_progress_label: bool = False
    completion_bonus: float = 0.0
    tau2_t3_progress_mode: str = "legacy"
    tau2_t3_min_turn: int = 6
    tau2_arew_label_mode: str = "clean"
    tau2_arew_min_turn: int = 8


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        npc_rollout_wg = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.npc_rollout_wg = npc_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        if self.config.dataset == 'MovieRec':
            # for unsupervised manner (See appendix D3 of the paper)
            self.epsilon_win = 0.28
        self.max_rolling_context_length = max(self.config.max_prompt_length, self.config.max_start_length)

    def _build_rm_scores(self, responses: torch.Tensor, final_rewards: List[float]) -> torch.Tensor:
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        response_mask = self.tensor_fn.create_attention_mask(responses)
        valid_lengths = response_mask.sum(dim=1).tolist()
        for idx, (valid_length, reward) in enumerate(zip(valid_lengths, final_rewards, strict=True)):
            if valid_length > 0:
                reward_tensor[idx, int(valid_length) - 1] = float(reward)
        return reward_tensor

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _batch_tokenize_(self, responses: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of responses. Mark -1 at the end of each response turn."""
        tokenized = self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        modified_input_ids = input_ids.clone()
        last_nonpad_idx = (attention_mask.sum(dim=1) - 1).long()
        modified_input_ids[torch.arange(input_ids.size(0)), last_nonpad_idx] = -1

        modified_input_ids = modified_input_ids.detach()

        return input_ids, modified_input_ids


    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at interact operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        responses_str = [_truncate_response_to_action(resp) for resp in responses_str]

        responses, modified_responses = self._batch_tokenize_(responses_str)
        return responses, modified_responses, responses_str

    def _postprocess_npc_responses(self, responses: torch.Tensor):
        """Process responses to stop at interact operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
        return responses_str

    def _build_arew_label_tensor(self, responses: torch.Tensor, step_labels: List[int]) -> torch.Tensor:
        """Broadcast one step label to all non-pad tokens of the current response."""
        labels = torch.zeros_like(responses, dtype=torch.int8)
        if len(step_labels) == 0:
            return labels
        response_mask = self.tensor_fn.create_attention_mask(responses).bool()
        step_label_tensor = torch.tensor(step_labels, device=responses.device, dtype=torch.int8).unsqueeze(1)
        return torch.where(response_mask, step_label_tensor.expand_as(labels), labels)

    def _build_arew_step_id_tensor(self, responses: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Broadcast one rollout-step id to all non-pad tokens of the current response."""
        step_ids = torch.zeros_like(responses, dtype=torch.int16)
        response_mask = self.tensor_fn.create_attention_mask(responses).bool()
        current_step_id = torch.full(
            (responses.size(0), 1),
            step_idx + 1,
            device=responses.device,
            dtype=torch.int16,
        )
        return torch.where(response_mask, current_step_id.expand_as(step_ids), step_ids)

    def _build_arew_flag_tensor(self, responses: torch.Tensor, step_flags: List[int | bool]) -> torch.Tensor:
        """Broadcast a per-step complementary fallback flag to all non-pad tokens."""
        flags = torch.zeros_like(responses, dtype=torch.int8)
        if len(step_flags) == 0:
            return flags
        response_mask = self.tensor_fn.create_attention_mask(responses).bool()
        step_flag_tensor = torch.tensor(
            [1 if bool(flag) else 0 for flag in step_flags],
            device=responses.device,
            dtype=torch.int8,
        ).unsqueeze(1)
        return torch.where(response_mask, step_flag_tensor.expand_as(flags), flags)

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        max_len = min(self.max_rolling_context_length, int(new_attention_mask.sum(dim=1).max().item()))

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_step: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                prompt_arew_labels: torch.Tensor,
                prompt_arew_step_ids: torch.Tensor,
                prompt_arew_complementary_mask: torch.Tensor,
                response: torch.Tensor, 
                response_with_step: torch.Tensor, 
                response_arew_labels: torch.Tensor,
                response_arew_step_ids: torch.Tensor,
                response_arew_complementary_mask: torch.Tensor,
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id

        tensors = [prompt, response]
        tensors_with_step = [prompt_with_step, response_with_step]
        tensors_with_mask = [prompt_with_mask, response]
        tensors_with_arew = [prompt_arew_labels, response_arew_labels]
        tensors_with_arew_step_ids = [prompt_arew_step_ids, response_arew_step_ids]
        tensors_with_arew_complementary = [prompt_arew_complementary_mask, response_arew_complementary_mask]
        
        if info is not None:
            tensors.append(info)
            tensors_with_step.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
            tensors_with_arew.append(torch.zeros(info.size(), dtype=prompt_arew_labels.dtype, device=info.device))
            tensors_with_arew_step_ids.append(
                torch.zeros(info.size(), dtype=prompt_arew_step_ids.dtype, device=info.device)
            )
            tensors_with_arew_complementary.append(
                torch.zeros(info.size(), dtype=prompt_arew_complementary_mask.dtype, device=info.device)
            )
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_step = torch.cat(tensors_with_step, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        concatenated_with_arew = torch.cat(tensors_with_arew, dim=1)
        concatenated_with_arew_step_ids = torch.cat(tensors_with_arew_step_ids, dim=1)
        concatenated_with_arew_complementary = torch.cat(tensors_with_arew_complementary, dim=1)

        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_step = concatenated_with_step.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)
        padded_tensor_with_arew = concatenated_with_arew.gather(1, sorted_indices)
        padded_tensor_with_arew_step_ids = concatenated_with_arew_step_ids.gather(1, sorted_indices)
        padded_tensor_with_arew_complementary = concatenated_with_arew_complementary.gather(1, sorted_indices)

        return (
            padded_tensor,
            padded_tensor_with_step,
            padded_tensor_with_info,
            padded_tensor_with_arew,
            padded_tensor_with_arew_step_ids,
            padded_tensor_with_arew_complementary,
        )

    def _update_right_side(self, right_side: Dict, 
                          cur_res_ids: torch.Tensor,
                          cur_res_ids_with_step: torch.Tensor,
                          cur_res_arew_labels: torch.Tensor,
                          cur_res_arew_step_ids: torch.Tensor,
                          cur_res_arew_complementary_mask: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            (
                responses,
                responses_with_step,
                responses_with_info_mask,
                responses_arew_labels,
                responses_arew_step_ids,
                responses_arew_complementary_mask,
            ) = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_step'],
                    right_side['responses_with_info_mask'],
                    right_side['responses_arew_labels'],
                    right_side['responses_arew_step_ids'],
                    right_side['responses_arew_complementary_mask'],
                    cur_res_ids,
                    cur_res_ids_with_step,
                    cur_res_arew_labels,
                    cur_res_arew_step_ids,
                    cur_res_arew_complementary_mask,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            (
                responses,
                responses_with_step,
                responses_with_info_mask,
                responses_arew_labels,
                responses_arew_step_ids,
                responses_arew_complementary_mask,
            ) = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_step'],
                    right_side['responses_with_info_mask'],
                    right_side['responses_arew_labels'],
                    right_side['responses_arew_step_ids'],
                    right_side['responses_arew_complementary_mask'],
                    cur_res_ids,
                    cur_res_ids_with_step,
                    cur_res_arew_labels,
                    cur_res_arew_step_ids,
                    cur_res_arew_complementary_mask,
                    pad_to_left=False
                )
        max_len = int(self.tensor_fn.create_attention_mask(responses).sum(dim=1).max().item())

        return {'responses': responses[:, :max_len],
                'responses_with_step': responses_with_step[:, :max_len],
                'responses_with_info_mask': responses_with_info_mask[:, :max_len],
                'responses_arew_labels': responses_arew_labels[:, :max_len],
                'responses_arew_step_ids': responses_arew_step_ids[:, :max_len],
                'responses_arew_complementary_mask': responses_arew_complementary_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto, rollout_wg) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, validate: bool = False, global_step: int = 0) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 
                               'responses_with_info_mask': initial_input_ids[:, []],
                               'responses_with_step': initial_input_ids[:, []],
                               'responses_arew_labels': initial_input_ids[:, []].to(torch.int8),
                               'responses_arew_step_ids': initial_input_ids[:, []].to(torch.int16),
                               'responses_arew_complementary_mask': initial_input_ids[:, []].to(torch.int8)}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        
        if self.config.dataset == "CircuitDecoding":
            controllers = [itm for itm in gen_batch.non_tensor_batch['controller']]
            search_spaces = [CircuitDecodingSpace(controller, global_step, self.config.trunc_strength) for controller in controllers]
        elif self.config.dataset == "MovieRec":
            controllers = [itm for itm in gen_batch.non_tensor_batch['controller']]
            search_spaces = [MovieRecSpace(controller, self.config.max_turns, global_step, self.config.trunc_strength, self.epsilon_win) for controller in controllers]
        elif self.config.dataset == "Tau2Bench":
            controllers = [itm for itm in gen_batch.non_tensor_batch['controller']]
            search_spaces = []
            standard_search_spaces = []
            effective_trunc_strength = self.config.trunc_strength if self.config.early_cut else None
            for controller in controllers:
                if controller.get("mode", "solo") == "standard":
                    space = Tau2StandardSpace(
                        controller,
                        self.config.max_turns,
                        global_step,
                        effective_trunc_strength,
                        self.config.hard_tolerate_num,
                        self.config.strict_progress_label,
                        self.config.completion_bonus,
                        self.config.tau2_t3_progress_mode,
                        self.config.tau2_t3_min_turn,
                        self.config.tau2_arew_label_mode,
                        self.config.tau2_arew_min_turn,
                        defer_bootstrap=True,
                    )
                    standard_search_spaces.append(space)
                else:
                    space = Tau2SoloSpace(
                        controller,
                        self.config.max_turns,
                        global_step,
                        effective_trunc_strength,
                        self.config.hard_tolerate_num,
                        self.config.strict_progress_label,
                    )
                search_spaces.append(space)
            if standard_search_spaces:
                max_workers = min(len(standard_search_spaces), _TAU2_STANDARD_USER_SIM_PARALLELISM)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    list(executor.map(lambda space: space.bootstrap_conversation(), standard_search_spaces))
        else:
            raise NotImplementedError(f"Unsupported dataset: {self.config.dataset}")
        rollings = gen_batch

        if self.config.dataset == "Tau2Bench":
            bootstrap_turns = [
                space.bootstrap_visible_turns() if hasattr(space, "bootstrap_visible_turns") else []
                for space in search_spaces
            ]
            if any(bootstrap_turns):
                bootstrap_obs = [_render_standard_bootstrap(turns) for turns in bootstrap_turns]
                bootstrap_obs_ids = self._process_next_obs(bootstrap_obs)
                rollings = self._update_rolling_state(
                    rollings,
                    initial_input_ids[:, []],
                    bootstrap_obs_ids,
                )
                original_left_side = {'input_ids': rollings.batch['input_ids'][:, -self.config.max_start_length:]}
            active_mask = torch.tensor([space.must_stop != 1.0 for space in search_spaces], dtype=torch.bool)
            turns_stats = active_mask.to(torch.int)

        active_num_list = [active_mask.sum().item()]
        meta_info = {}

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })       
            # print(rollings_active.batch['input_ids'][0])     
            gen_output = self._generate_with_gpu_padding(rollings_active, rollout_wg=self.actor_rollout_wg)

            meta_info = gen_output.meta_info            
            responses_ids, responses_ids_with_step, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_ids_with_step, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_ids_with_step, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search, step_labels, step_fallback_flags = self.execute_predictions(
                responses_str, search_spaces, self.tokenizer.pad_token, active_mask, validate, global_step
            )
            cur_res_arew_labels = self._build_arew_label_tensor(responses_ids, step_labels)
            cur_res_arew_step_ids = self._build_arew_step_id_tensor(responses_ids, step)
            cur_res_arew_complementary_mask = self._build_arew_flag_tensor(responses_ids, step_fallback_flags)
            
            # i-th turn finished. Update as follows.
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            if step < self.config.max_turns - 1:
                next_obs_ids = self._process_next_obs(next_obs)
                
                # Update states
                rollings = self._update_rolling_state(
                    rollings,
                    responses_ids,
                    next_obs_ids
                )
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    responses_ids_with_step,
                    cur_res_arew_labels,
                    cur_res_arew_step_ids,
                    cur_res_arew_complementary_mask,
                    next_obs_ids
                )
            else:
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    responses_ids_with_step,
                    cur_res_arew_labels,
                    cur_res_arew_step_ids,
                    cur_res_arew_complementary_mask
                )
        
        if active_mask.sum():
            # We may expect `dones` to be all-ones after the final turn.
            print("WARNING: Still active examples after final rollout.")

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        meta_info['early_cut'] = [sp.cut for sp in search_spaces]
        meta_info['repeats'] = [float(sp.repeat_counts / (sp.query_counts - 1)) if sp.query_counts > 1 else 0 for sp in search_spaces]
        meta_info['stalling'] = [float(sp.stalling_counts / (sp.query_counts -1)) if sp.query_counts > 1 else 0 for sp in search_spaces]
        meta_info['must_stop'] = [sp.must_stop for sp in search_spaces]
        if self.config.dataset == "Tau2Bench":
            meta_info['tau2_invalid_actions'] = [sp.invalid_action_count for sp in search_spaces]
            meta_info['tau2_tool_errors'] = [sp.tool_error_count for sp in search_spaces]
            meta_info['tau2_no_state_change_writes'] = [sp.no_state_change_write_count for sp in search_spaces]
            meta_info['tau2_hard_truncations'] = [sp.hard_truncation for sp in search_spaces]
            meta_info['tau2_soft_truncations'] = [sp.soft_truncation for sp in search_spaces]
            meta_info['tau2_positive_steps'] = [sum(1 for label in sp.step_label_history if label > 0) for sp in search_spaces]
            meta_info['tau2_neutral_steps'] = [sum(1 for label in sp.step_label_history if label == 0) for sp in search_spaces]
            meta_info['tau2_negative_steps'] = [sum(1 for label in sp.step_label_history if label < 0) for sp in search_spaces]
            meta_info['tau2_arew_positive_steps'] = [
                sum(1 for label in getattr(sp, "arew_label_history", []) if label > 0) for sp in search_spaces
            ]
            meta_info['tau2_arew_neutral_steps'] = [
                sum(1 for label in getattr(sp, "arew_label_history", []) if label == 0) for sp in search_spaces
            ]
            meta_info['tau2_arew_negative_steps'] = [
                sum(1 for label in getattr(sp, "arew_label_history", []) if label < 0) for sp in search_spaces
            ]
            meta_info['tau2_arew_has_pos_neg'] = [
                float(
                    any(label > 0 for label in getattr(sp, "arew_label_history", []))
                    and any(label < 0 for label in getattr(sp, "arew_label_history", []))
                )
                for sp in search_spaces
            ]
            meta_info['tau2_message_turns'] = [getattr(sp, "assistant_message_turn_count", 0) for sp in search_spaces]
            meta_info['tau2_tool_turns'] = [getattr(sp, "assistant_tool_turn_count", 0) for sp in search_spaces]
            meta_info['tau2_read_tool_turns'] = [getattr(sp, "assistant_read_tool_turn_count", 0) for sp in search_spaces]
            meta_info['tau2_write_tool_turns'] = [getattr(sp, "assistant_write_tool_turn_count", 0) for sp in search_spaces]
            meta_info['tau2_user_tool_hops'] = [getattr(sp, "user_tool_hop_count", 0) for sp in search_spaces]
            meta_info['tau2_bootstrap_user_tool_hops'] = [getattr(sp, "bootstrap_user_tool_hop_count", 0) for sp in search_spaces]
            meta_info['tau2_user_disconnects'] = [getattr(sp, "user_disconnect_count", 0) for sp in search_spaces]
            meta_info['tau2_bootstrap_user_disconnects'] = [getattr(sp, "bootstrap_user_disconnect_count", 0) for sp in search_spaces]
            meta_info['tau2_disconnect_repeat_neutral_steps'] = [
                getattr(sp, "disconnect_repeat_neutral_count", 0) for sp in search_spaces
            ]
            meta_info['tau2_assistant_tool_errors'] = [getattr(sp, "assistant_tool_error_count", 0) for sp in search_spaces]
            meta_info['tau2_user_tool_errors'] = [getattr(sp, "user_tool_error_count", 0) for sp in search_spaces]
            meta_info['tau2_state_changed_steps'] = [getattr(sp, "state_changed_step_count", 0) for sp in search_spaces]
            meta_info['tau2_action_progress_steps'] = [getattr(sp, "action_progress_step_count", 0) for sp in search_spaces]
            meta_info['tau2_new_raw_information_steps'] = [getattr(sp, "new_raw_information_step_count", 0) for sp in search_spaces]
            meta_info['tau2_new_normalized_information_steps'] = [
                getattr(sp, "new_normalized_information_step_count", 0) for sp in search_spaces
            ]
            meta_info['tau2_new_family_steps'] = [getattr(sp, "new_family_step_count", 0) for sp in search_spaces]
            meta_info['tau2_max_no_progress_streak'] = [getattr(sp, "max_soft_no_progress_streak", 0) for sp in search_spaces]
            meta_info['tau2_t3_truncation_turn'] = [getattr(sp, "t3_truncation_turn", 0) for sp in search_spaces]
            meta_info['tau2_user_stopped'] = [1.0 if getattr(sp, "user_stopped", False) else 0.0 for sp in search_spaces]

        if self.config.dataset in ['MovieRec']:
            meta_info['consecutive_drop_counts'] = [sp.consecutive_drop_counts for sp in search_spaces]

        final_rewards = []
        tau2_official_rewards = None
        tau2_train_fractional_rewards = None
        tau2_stop_reasons = None
        if self.config.dataset == "Tau2Bench":
            if validate:
                tau2_official_rewards = [space.finalize(official=True).reward for space in search_spaces]
                tau2_train_fractional_rewards = [space.finalize(official=False).reward for space in search_spaces]
                final_rewards = tau2_official_rewards
            else:
                tau2_train_fractional_rewards = [space.finalize(official=False).reward for space in search_spaces]
                final_rewards = tau2_train_fractional_rewards
            tau2_stop_reasons = [
                getattr(space, "user_stop_reason", None) or "max_turn_or_runtime_end"
                for space in search_spaces
            ]
            meta_info['tau2_final_rewards'] = final_rewards

        meta_info['active_trajectory_counts'] = active_num_list
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info, gen_batch.non_tensor_batch)
        if self.config.dataset == "Tau2Bench":
            final_output.batch["rm_scores"] = self._build_rm_scores(final_output.batch["responses"], final_rewards)
            final_output.batch["responses_arew_labels"] = final_output.batch["responses_arew_labels"].to(torch.int8)
            final_output.batch["responses_arew_step_ids"] = final_output.batch["responses_arew_step_ids"].to(torch.int16)
            final_output.batch["responses_arew_complementary_mask"] = final_output.batch[
                "responses_arew_complementary_mask"
            ].to(torch.int8)
            final_output.non_tensor_batch["final_rewards"] = np.array(final_rewards, dtype=np.float32)
            if tau2_official_rewards is not None:
                final_output.non_tensor_batch["tau2_official_rewards"] = np.array(tau2_official_rewards, dtype=np.float32)
            if tau2_train_fractional_rewards is not None:
                final_output.non_tensor_batch["tau2_train_fractional_rewards"] = np.array(
                    tau2_train_fractional_rewards,
                    dtype=np.float32,
                )
            if tau2_stop_reasons is not None:
                final_output.non_tensor_batch["tau2_stop_reason"] = np.array(tau2_stop_reasons, dtype=object)
            final_output.non_tensor_batch["tau2_task_id"] = np.array([space.task_id for space in search_spaces], dtype=object)
            final_output.non_tensor_batch["tau2_trajectory"] = np.array(
                [space.format_trajectory() for space in search_spaces],
                dtype=object,
            )
        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            non_tensor_dict: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['response_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output, non_tensors=non_tensor_dict)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], search_spaces, pad_token: str, active_mask=None, validate=False, global_step: int=2) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        if not validate and not self.config.early_cut:
            validate = True
        allow_invalid_retry = validate or self.config.dataset == "Tau2Bench"

        del pad_token
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search, step_labels, step_fallback_flags = [], [], [], [], [], []
        
        assert len(contents) == len(search_spaces), f"#Interacts != #GroundTruths: {len(contents)} != {len(search_spaces)}"
        
        if self.config.dataset in ['CircuitDecoding', 'MovieRec', 'Tau2Bench']:
            interact_payloads = [
                (content, search_space)
                for action, content, search_space, active in zip(cur_actions, contents, search_spaces, active_mask)
                if bool(active) and action == 'interact'
            ]
            message_payloads = [
                (content, search_space)
                for action, content, search_space, active in zip(cur_actions, contents, search_spaces, active_mask)
                if bool(active) and action == 'message' and hasattr(search_space, "compute_message_feedback")
            ]
            feedback_results = [
                search_space.compute_feedback(content, validate)
                for content, search_space in interact_payloads
            ]
            if self.config.dataset == "Tau2Bench" and len(message_payloads) > 1:
                max_workers = min(len(message_payloads), _TAU2_STANDARD_USER_SIM_PARALLELISM)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    message_feedback_results = list(
                        executor.map(
                            lambda item: item[1].compute_message_feedback(item[0], validate),
                            message_payloads,
                        )
                    )
            else:
                message_feedback_results = [
                    search_space.compute_message_feedback(content, validate)
                    for content, search_space in message_payloads
                ]
        else:
            feedback_results = []
            message_feedback_results = []
        
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                step_labels.append(0)
                step_fallback_flags.append(0)
            else:
                if action == 'answer':
                    if getattr(search_spaces[i], "supports_answer", True):
                        if self.config.dataset == "Tau2Bench":
                            search_spaces[i].register_answer(contents[i])
                        valid_action_value = 1
                    else:
                        search_spaces[i].register_answer(contents[i])
                        valid_action_value = 0
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(valid_action_value)
                    is_search.append(0)
                    if self.config.dataset == "Tau2Bench" and hasattr(search_spaces[i], "last_arew_label"):
                        step_labels.append(search_spaces[i].last_arew_label)
                    elif self.config.dataset == "Tau2Bench":
                        step_labels.append(search_spaces[i].last_step_label)
                    else:
                        step_labels.append(0)
                    step_fallback_flags.append(0)
                elif action == 'interact':
                    feedbacks = feedback_results.pop(0)
                    step_label = 0
                    step_fallback_flag = 0
                    if self.config.dataset == "Tau2Bench":
                        if hasattr(search_spaces[i], "last_arew_label"):
                            step_label = search_spaces[i].last_arew_label
                            step_fallback_flag = int(getattr(search_spaces[i], "last_arew_fallback_eligible", 0))
                        else:
                            step_label = search_spaces[i].last_step_label
                    if feedbacks:
                        if isinstance(feedbacks, dict):
                            next_obs.append(_render_chat_observation(feedbacks["role"], feedbacks["content"]))
                        else:
                            next_obs.append(
                                f"{start_user}"
                                f"The feedback of your latest interaction: {feedbacks.strip()}\n"
                                f"Now it is your turn:\n"
                                f"{end_signal}"
                                f"{start_assistant}"
                            )
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(1)
                    else:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                        is_search.append(1)
                    step_labels.append(step_label)
                    step_fallback_flags.append(step_fallback_flag)
                elif action == 'message' and hasattr(search_spaces[i], "compute_message_feedback"):
                    feedbacks = message_feedback_results.pop(0)
                    step_label = 0
                    step_fallback_flag = 0
                    if self.config.dataset == "Tau2Bench":
                        if hasattr(search_spaces[i], "last_arew_label"):
                            step_label = search_spaces[i].last_arew_label
                            step_fallback_flag = int(getattr(search_spaces[i], "last_arew_fallback_eligible", 0))
                        else:
                            step_label = search_spaces[i].last_step_label
                    if feedbacks:
                        if isinstance(feedbacks, dict):
                            next_obs.append(_render_chat_observation(feedbacks["role"], feedbacks["content"]))
                        else:
                            next_obs.append(
                                f"{start_user}"
                                f"{feedbacks.strip()}\n"
                                f"{end_signal}"
                                f"{start_assistant}"
                            )
                        dones.append(0)
                        valid_action.append(1)
                        is_search.append(0)
                    else:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(1)
                        is_search.append(0)
                    step_labels.append(step_label)
                    step_fallback_flags.append(step_fallback_flag)
                else:
                    if self.config.dataset == "Tau2Bench":
                        search_spaces[i].register_invalid_action("invalid_prediction_format")
                    think_retry_hint = ""
                    if self.config.dataset == "Tau2Bench" and search_spaces[i].controller.get("enable_think", False):
                        if getattr(search_spaces[i], "supports_answer", True):
                            think_retry_hint = (
                                " If think mode is enabled, you should first write a short <think>...</think> block and then "
                                "output either one <interact>...</interact> or one <answer>...</answer> block."
                            )
                        else:
                            think_retry_hint = (
                                " If think mode is enabled, you should first write a short <think>...</think> block and then "
                                "output either one <message>...</message> or one <interact>...</interact> block."
                            )
                    if allow_invalid_retry:
                        if getattr(search_spaces[i], "supports_answer", True):
                            invalid_instruction = (
                                "The previous action is invalid. If you want to make an interaction, you should put it between "
                                "<interact> and </interact>. If you want to give the final answer, you should put the answer "
                                "between <answer> and </answer>."
                            )
                        else:
                            invalid_instruction = (
                                "The previous action is invalid. If you want to speak to the user, you should put it between "
                                "<message> and </message>. If you want to make an interaction, you should put it between "
                                "<interact> and </interact>."
                            )
                        next_obs.append(
                            f"{start_user}"
                            f"{invalid_instruction}{think_retry_hint} Try again.\n"
                            f"{end_signal}"
                            f"{start_assistant}"
                        )                        
                    # next_obs.append(f'\n\n\n\n')
                        dones.append(0)
                        valid_action.append(0)
                        is_search.append(0)
                    else:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(0)
                        is_search.append(0)
                    step_labels.append(-1)
                    step_fallback_flags.append(0)
            
        assert len(feedback_results) == 0
        assert len(message_feedback_results) == 0
            
        return next_obs, dones, valid_action, is_search, step_labels, step_fallback_flags

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(interact|answer|message)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents
