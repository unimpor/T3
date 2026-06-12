import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from search_r1.arew_tasks.spaces import FloDialSpace, MediQSpace, PreferenceEstimationSpace
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from verl import DataProto


AREW_DATA2KEYS = {
    "PE-G": ["controller"],
    "PE-F": ["controller"],
    "MediQ": ["controller"],
    "FloDial": ["controller"],
}


def is_arew_task(dataset: str) -> bool:
    return dataset in AREW_DATA2KEYS


@dataclass
class AREWGenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    dataset: str
    experiment_name: str = ""
    as_bonus: bool = False
    bt_bonus: bool = False
    as_cf: bool = False
    bt_cf: bool = False
    btv: str = "v1"


class AREWGenerationManager(LLMGenerationManager):
    """Generation loop for AREW preference/medical/dialogue tasks.

    The base T3 manager owns tensor packing and token-level AREW label tensors.
    This subclass keeps the dataset-specific alternating AS/BT interaction loop.
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: AREWGenerationConfig,
        is_validation: bool = False,
        npc_rollout_wg=None,
    ):
        base_config = GenerationConfig(
            max_turns=config.max_turns,
            max_start_length=config.max_start_length,
            max_prompt_length=config.max_prompt_length,
            max_response_length=config.max_response_length,
            max_obs_length=config.max_obs_length,
            num_gpus=config.num_gpus,
            dataset=config.dataset,
        )
        super().__init__(
            tokenizer=tokenizer,
            actor_rollout_wg=actor_rollout_wg,
            config=base_config,
            is_validation=is_validation,
            npc_rollout_wg=npc_rollout_wg,
        )
        self.arew_config = config
        if config.max_turns % 2 != 0:
            raise ValueError("AREW PE/MediQ/FloDial rollouts require an even max_turns value.")

        experiment_name = config.experiment_name.lower()
        if "llama" in experiment_name:
            self.start_user = "<|start_header_id|>user<|end_header_id|>\n\n"
            self.start_assistant = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            self.end_signal = "<|eot_id|>"
        else:
            self.start_user = "\n<|im_start|>user\n"
            self.start_assistant = "\n<|im_start|>assistant\n"
            self.end_signal = "<|im_end|>"

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        responses_str = [self._truncate_to_action(resp) for resp in responses_str]
        responses, modified_responses = self._batch_tokenize_(responses_str)
        return responses, modified_responses, responses_str

    def _render_observation(self, content: str) -> str:
        return f"{self.start_user}{content.strip()}\n{self.end_signal}{self.start_assistant}"

    def _pack_user_sim_batch(self, prompts: List[str]) -> DataProto:
        tokenized = self.tokenizer(
            [self._render_observation(prompt) for prompt in prompts],
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = self.tensor_fn.create_position_ids(attention_mask)
        return DataProto.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            meta_info={
                "recompute_log_prob": False,
                "do_sample": False,
            },
        )

    def run_llm_loop(
        self,
        gen_batch,
        initial_input_ids: torch.Tensor,
        validate: bool = False,
        global_step: int = 0,
    ) -> DataProto:
        del global_step

        original_left_side = {"input_ids": initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            "responses": initial_input_ids[:, []],
            "responses_with_info_mask": initial_input_ids[:, []],
            "responses_with_step": initial_input_ids[:, []],
            "responses_arew_labels": initial_input_ids[:, []].to(torch.int8),
            "responses_arew_step_ids": initial_input_ids[:, []].to(torch.int16),
            "responses_arew_complementary_mask": initial_input_ids[:, []].to(torch.int8),
        }

        controllers = [item for item in gen_batch.non_tensor_batch["controller"]]
        search_spaces = self._build_search_spaces(controllers)

        active_mask = torch.ones(gen_batch.batch["input_ids"].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        as_positive_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        as_negative_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        bt_positive_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)
        bt_negative_stats = torch.zeros(gen_batch.batch["input_ids"].shape[0], dtype=torch.int)

        active_num_list = [active_mask.sum().item()]
        meta_info: Dict[str, Any] = {}
        rollings = gen_batch
        rollings_cf = None

        for step in range(1, self.config.max_turns + 1):
            if not active_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=["input_ids", "attention_mask", "position_ids"],
            )
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})

            predictions_cf = None
            if self._should_generate_cf_response(step, rollings_cf, validate):
                rollings_cf.batch = self.tensor_fn.cut_to_effective_len(
                    rollings_cf.batch,
                    keys=["input_ids", "attention_mask", "position_ids"],
                )
                rollings_cf_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings_cf.batch.items()})
                gen_output_cf = self._generate_with_gpu_padding(rollings_cf_active, rollout_wg=self.actor_rollout_wg)
                _, _, predictions_cf_active = self._postprocess_responses(gen_output_cf.batch["responses"])
                predictions_cf = self._pad_active_strings(predictions_cf_active, active_mask)

            gen_output = self._generate_with_gpu_padding(rollings_active, rollout_wg=self.actor_rollout_wg)
            meta_info = gen_output.meta_info
            responses_ids, responses_ids_with_step, responses_str = self._postprocess_responses(
                gen_output.batch["responses"]
            )
            responses_ids, responses_ids_with_step, responses_str = self.tensor_fn._example_level_pad(
                responses_ids,
                responses_ids_with_step,
                responses_str,
                active_mask,
            )

            next_obs, dones, valid_action, is_search, step_labels = self.execute_arew_predictions(
                predictions=responses_str,
                search_spaces=search_spaces,
                active_mask=active_mask,
                validate=validate,
                step=step,
                predictions_cf=predictions_cf,
            )

            self._update_label_stats(
                step,
                step_labels,
                active_mask,
                as_positive_stats,
                as_negative_stats,
                bt_positive_stats,
                bt_negative_stats,
            )
            cur_res_arew_labels = self._build_arew_label_tensor(responses_ids, step_labels)
            cur_res_arew_step_ids = self._build_arew_step_id_tensor(responses_ids, step - 1)
            cur_res_arew_complementary_mask = self._build_arew_flag_tensor(responses_ids, [0] * len(step_labels))

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            if step < self.config.max_turns:
                next_obs_ids = self._process_next_obs(next_obs)
                rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
                rollings_cf = self._maybe_update_counterfactual_rollings(
                    rollings=rollings,
                    responses_ids=responses_ids,
                    active_mask=active_mask,
                    search_spaces=search_spaces,
                    step=step,
                    validate=validate,
                )
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    responses_ids_with_step,
                    cur_res_arew_labels,
                    cur_res_arew_step_ids,
                    cur_res_arew_complementary_mask,
                    next_obs_ids,
                )
            else:
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    responses_ids_with_step,
                    cur_res_arew_labels,
                    cur_res_arew_step_ids,
                    cur_res_arew_complementary_mask,
                )

        if active_mask.sum():
            print("WARNING: Still active examples after final rollout.")

        meta_info["turns_stats"] = turns_stats.tolist()
        meta_info["active_mask"] = active_mask.tolist()
        meta_info["valid_action_stats"] = valid_action_stats.tolist()
        meta_info["valid_search_stats"] = valid_search_stats.tolist()
        meta_info["early_cut"] = [getattr(space, "cut", 0.0) for space in search_spaces]
        meta_info["arew_as_positive_steps"] = as_positive_stats.tolist()
        meta_info["arew_as_negative_steps"] = as_negative_stats.tolist()
        meta_info["arew_bt_positive_steps"] = bt_positive_stats.tolist()
        meta_info["arew_bt_negative_steps"] = bt_negative_stats.tolist()
        meta_info["active_trajectory_counts"] = active_num_list

        if self.config.dataset in {"PE-G", "PE-F"}:
            meta_info["repeats"] = [
                float(space.repeat_counts / (space.query_counts - 1)) if space.query_counts > 1 else 0.0
                for space in search_spaces
            ]
            meta_info["stalling"] = [
                float(space.stalling_counts / (space.query_counts - 1)) if space.query_counts > 1 else 0.0
                for space in search_spaces
            ]
            meta_info["must_stop"] = [space.must_stop for space in search_spaces]
            meta_info["consecutive_drop_counts"] = [space.consecutive_drop_counts for space in search_spaces]

        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info, gen_batch.non_tensor_batch)
        final_output.batch["responses_arew_labels"] = final_output.batch["responses_arew_labels"].to(torch.int8)
        final_output.batch["responses_arew_step_ids"] = final_output.batch["responses_arew_step_ids"].to(torch.int16)
        final_output.batch["responses_arew_complementary_mask"] = final_output.batch[
            "responses_arew_complementary_mask"
        ].to(torch.int8)
        return final_output

    def execute_arew_predictions(
        self,
        predictions: List[str],
        search_spaces,
        active_mask: torch.Tensor,
        validate: bool,
        step: int,
        predictions_cf: List[str] | None = None,
    ) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
        actions, contents = self.postprocess_predictions(predictions)
        assert len(contents) == len(search_spaces), f"#Interacts != #GroundTruths: {len(contents)} != {len(search_spaces)}"

        if self.config.dataset in {"PE-G", "PE-F"}:
            feedback_results = self._execute_preference_predictions(actions, contents, search_spaces, active_mask, step)
        else:
            feedback_results = self._execute_user_sim_predictions(actions, contents, search_spaces, active_mask, step)

        if self.config.dataset == "MediQ" and predictions_cf is not None:
            self._record_counterfactual_predictions(predictions_cf, search_spaces, active_mask, step)

        step_labels = self._compute_step_labels(search_spaces, active_mask, actions, validate, step)
        next_obs, dones, valid_action, is_search = [], [], [], []

        for action, active in zip(actions, active_mask):
            if not bool(active):
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                continue

            if action == "answer":
                next_obs.append("")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                continue

            feedback = feedback_results.pop(0)
            if feedback:
                next_obs.append(self._render_observation(feedback))
                dones.append(0)
            else:
                next_obs.append("")
                dones.append(1)
            valid_action.append(1 if action == "interact" else 0)
            is_search.append(1 if action == "interact" else 0)

        assert len(feedback_results) == 0
        return next_obs, dones, valid_action, is_search, step_labels

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str | None], List[str]]:
        actions = []
        contents = []
        for prediction in predictions:
            if not isinstance(prediction, str):
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            match = re.search(r"<(interact|answer)>(.*?)</\1>", prediction, re.DOTALL)
            if match:
                actions.append(match.group(1))
                contents.append(match.group(2).strip())
            else:
                actions.append(None)
                contents.append("")
        return actions, contents

    def _build_search_spaces(self, controllers):
        if self.config.dataset == "PE-G":
            return [PreferenceEstimationSpace(controller, self.config.max_turns, "gated3") for controller in controllers]
        if self.config.dataset == "PE-F":
            return [PreferenceEstimationSpace(controller, self.config.max_turns, "full") for controller in controllers]
        if self.config.dataset == "MediQ":
            return [MediQSpace(controller, self.config.max_turns) for controller in controllers]
        if self.config.dataset == "FloDial":
            return [FloDialSpace(controller, self.config.max_turns, self.arew_config.btv) for controller in controllers]
        raise NotImplementedError(f"Unsupported AREW dataset: {self.config.dataset}")

    def _execute_preference_predictions(self, actions, contents, search_spaces, active_mask, step: int) -> List[str]:
        feedback_results = []
        for action, content, search_space, active in zip(actions, contents, search_spaces, active_mask):
            if not bool(active) or action == "answer":
                continue
            feedback_results.append(search_space.compute_feedback(content, step))
        return feedback_results

    def _execute_user_sim_predictions(self, actions, contents, search_spaces, active_mask, step: int) -> List[str]:
        feedback_raw = [None] * len(search_spaces)
        parsed_values = [None] * len(search_spaces)
        interact_data = []
        interact_positions = []

        for idx, (action, content, search_space, active) in enumerate(zip(actions, contents, search_spaces, active_mask)):
            if not bool(active) or action == "answer":
                continue
            value = search_space.extract_query(content, step) if action == "interact" else None
            parsed_values[idx] = value
            if value is not None and step % 2 == 1:
                interact_data.append((search_space, value))
                interact_positions.append(idx)

        for search_space, value, active in zip(search_spaces, parsed_values, active_mask):
            if bool(active):
                search_space.update_query(value, step)

        if interact_data:
            batch_queries = [search_space.generate_prompt(query) for search_space, query in interact_data]
            packed_batch = self._pack_user_sim_batch(batch_queries)
            rollout_wg = self.npc_rollout_wg or self.actor_rollout_wg
            batch_responses = self._generate_with_gpu_padding(packed_batch, rollout_wg=rollout_wg)
            interact_feedbacks = self._postprocess_npc_responses(batch_responses.batch["responses"])
            for idx, feedback in zip(interact_positions, interact_feedbacks):
                feedback_raw[idx] = feedback

        feedback_results = []
        for search_space, feedback, active, action in zip(search_spaces, feedback_raw, active_mask, actions):
            if not bool(active) or action == "answer":
                continue
            feedback_results.append(search_space.update_feedback(feedback, step))
        return feedback_results

    def _record_counterfactual_predictions(self, predictions_cf: List[str], search_spaces, active_mask, step: int) -> None:
        actions_cf, contents_cf = self.postprocess_predictions(predictions_cf)
        for action, content, search_space, active in zip(actions_cf, contents_cf, search_spaces, active_mask):
            if not bool(active):
                continue
            value = search_space.extract_query(content, step) if action in {"interact", "answer"} else None
            search_space.counterfactual(value, step)

    def _compute_step_labels(self, search_spaces, active_mask, actions, validate: bool, step: int) -> List[int]:
        step_labels = []
        for search_space, active, action in zip(search_spaces, active_mask, actions):
            if not bool(active) or action == "answer":
                step_labels.append(0)
            elif step % 2 == 1:
                step_labels.append(search_space.check_asinfo(validate) if self.arew_config.as_bonus else 0)
            else:
                step_labels.append(search_space.check_margin(validate) if self.arew_config.bt_bonus else 0)
        if step % 2 == 1:
            print("AS [0, 1, -1]: ", step_labels.count(0), step_labels.count(1), step_labels.count(-1))
        else:
            print("BT [0, 1, -1]: ", step_labels.count(0), step_labels.count(1), step_labels.count(-1))
        return step_labels

    def _should_generate_cf_response(self, step: int, rollings_cf, validate: bool) -> bool:
        if self.config.dataset != "MediQ" or validate or rollings_cf is None:
            return False
        if step % 2 == 0:
            return self.arew_config.bt_bonus and self.arew_config.bt_cf
        return self.arew_config.as_bonus and self.arew_config.as_cf

    def _maybe_update_counterfactual_rollings(
        self,
        rollings: DataProto,
        responses_ids: torch.Tensor,
        active_mask: torch.Tensor,
        search_spaces,
        step: int,
        validate: bool,
    ):
        if self.config.dataset != "MediQ" or validate:
            return None
        if not active_mask.sum():
            return None
        if step % 2 == 1 and step < self.config.max_turns - 1 and self.arew_config.bt_bonus and self.arew_config.bt_cf:
            next_obs_cf = [
                self._render_observation(
                    f"{space.counterfactual_feedback_observation()}\n"
                    "\nThe next round is **Guess Round**.\n"
                    "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about how the feedback supports or weakens hypotheses in <scratch> blocks. Update the state vector in\n"
                    "<interact>\n"
                    "Guess: wA,wB,wC,wD\n"
                    "</interact>\n"
                )
                if bool(active)
                else ""
                for space, active in zip(search_spaces, active_mask)
            ]
            next_obs_cf_ids = self._process_next_obs(next_obs_cf)
            return self._update_rolling_state(rollings, responses_ids, next_obs_cf_ids)

        if step % 2 == 0 and self.arew_config.as_bonus and self.arew_config.as_cf:
            responses_str_cf_active = [space.gen_counterfactual() + self.end_signal for space, active in zip(search_spaces, active_mask) if bool(active)]
            responses_ids_cf, responses_ids_with_step_cf = self._batch_tokenize_(responses_str_cf_active)
            responses_ids_cf, _, _ = self.tensor_fn._example_level_pad(
                responses_ids_cf,
                responses_ids_with_step_cf,
                responses_str_cf_active,
                active_mask,
            )
            next_obs_cf = [
                self._render_observation(
                    "\nThe next round is **Action Round**.\n"
                    "Briefly Reason (only essential reasoning in at most 2-3 sentences.) about what to ask in <scratch> blocks. Ask ONE atomic query to reduce uncertainty among the hypotheses in\n"
                    "<interact>\n"
                    "Query: ...\n"
                    "</interact>\n"
                )
                if bool(active)
                else ""
                for active in active_mask
            ]
            next_obs_cf_ids = self._process_next_obs(next_obs_cf)
            return self._update_rolling_state(rollings, responses_ids_cf, next_obs_cf_ids)
        return None

    def _update_label_stats(
        self,
        step: int,
        step_labels: List[int],
        active_mask: torch.Tensor,
        as_positive_stats: torch.Tensor,
        as_negative_stats: torch.Tensor,
        bt_positive_stats: torch.Tensor,
        bt_negative_stats: torch.Tensor,
    ) -> None:
        labels = torch.tensor(step_labels, dtype=torch.int)
        active = active_mask.to(torch.bool)
        if step % 2 == 1:
            as_positive_stats += ((labels > 0) & active).to(torch.int)
            as_negative_stats += ((labels < 0) & active).to(torch.int)
        else:
            bt_positive_stats += ((labels > 0) & active).to(torch.int)
            bt_negative_stats += ((labels < 0) & active).to(torch.int)

    def _pad_active_strings(self, active_strings: List[str], active_mask: torch.Tensor) -> List[str]:
        padded = [""] * int(active_mask.shape[0])
        cursor = 0
        for idx, active in enumerate(active_mask):
            if bool(active):
                padded[idx] = active_strings[cursor]
                cursor += 1
        return padded

    def _truncate_to_action(self, response: str) -> str:
        closing_tags = ["</answer>", "</interact>"]
        matches = [(response.find(tag), tag) for tag in closing_tags if response.find(tag) != -1]
        if not matches:
            return response
        idx, tag = min(matches, key=lambda item: item[0])
        return response[: idx + len(tag)] + self.end_signal
