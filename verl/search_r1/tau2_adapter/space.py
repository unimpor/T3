from copy import deepcopy
import json

from search_r1.tau2_adapter.action_parser import parse_action_string
from search_r1.tau2_adapter.core.message import AssistantMessage
from search_r1.tau2_adapter.core.toolkit import ToolType
from search_r1.tau2_adapter.eval.action import extract_tool_calls
from search_r1.tau2_adapter.eval.evaluator import evaluate_task
from search_r1.tau2_adapter.loader.registry import get_env_constructor
from search_r1.tau2_adapter.loader.tasks import get_tasks
from search_r1.tau2_adapter.prompts import build_solo_prompt


class Tau2SoloSpace:
    supports_answer = True

    def __init__(
        self,
        controller: dict,
        max_turns: int,
        global_step: int = 0,
        trunc_strength=None,
        hard_tolerate_num: int = 2,
        strict_progress_label: bool = False,
    ):
        del global_step
        self.controller = controller
        self.domain = controller["domain"]
        self.task_split = controller.get("task_split")
        self.task_pool = controller.get("task_pool")
        self.task_id = controller["task_id"]
        self.reward_mode = controller.get("reward_mode", "action")
        self.env_kwargs = controller.get("env_kwargs", {})
        self.cut = 0.0
        self.must_stop = 0.0
        self.query_counts = 0
        self.repeat_counts = 0
        self.stalling_counts = 0
        self.invalid_action_count = 0
        self.tool_error_count = 0
        self.no_state_change_write_count = 0
        self.hard_truncation = 0
        self.soft_truncation = 0
        self.max_counts = max_turns - 1
        self.all_queries = []
        self.final_reward = 0.0
        self.reward_info = None
        self.tolerate_num = int(trunc_strength) if trunc_strength is not None else 0
        self.hard_tolerate_num = int(hard_tolerate_num)
        self.strict_progress_label = bool(strict_progress_label)
        self.hard_bad_streak = 0
        self.soft_no_progress_streak = 0
        self.prev_matched_action_count = 0
        self.seen_observation_signatures = set()
        self.proxy_reason_history = []
        self.last_step_label = 0
        self.last_step_reason = None
        self.step_label_history = []

        lookup_task_split = None if self.task_pool is not None else self.task_split
        tasks = get_tasks(self.domain, task_split_name=lookup_task_split, task_ids=[self.task_id], solo_only=True)
        self.task = tasks[0]
        self.env_constructor = get_env_constructor(self.domain)
        self.env = self.env_constructor(solo_mode=True, **self.env_kwargs)
        self.initial_history = []
        if self.task.initial_state is not None:
            self.initial_history = deepcopy(self.task.initial_state.message_history or [])
            self.env.set_state(
                initialization_data=self.task.initial_state.initialization_data,
                initialization_actions=self.task.initial_state.initialization_actions,
                message_history=self.initial_history,
            )
        self.trajectory = []
        self.last_agent_db_hash = self.env.get_db_hash()
        self.last_user_db_hash = self.env.get_user_db_hash()

    def build_prompt(self) -> str:
        return build_solo_prompt(
            self.task,
            self.env,
            enable_think=self.controller.get("enable_think", False),
            think_mode=self.controller.get("think_mode", "short"),
        )

    def _get_tool_type(self, tool_name: str) -> ToolType:
        if self.env.tools is not None and self.env.tools.has_tool(tool_name):
            return self.env.tools.tool_type(tool_name)
        if self.env.user_tools is not None and self.env.user_tools.has_tool(tool_name):
            return self.env.user_tools.tool_type(tool_name)
        return ToolType.GENERIC

    def _get_env_state_signature(self) -> tuple[str | None, str | None]:
        return self.env.get_db_hash(), self.env.get_user_db_hash()

    def _matched_action_count(self) -> int:
        actions = getattr(self.task.evaluation_criteria, "actions", None)
        if not actions:
            return 0
        full_trajectory = [*self.initial_history, *self.trajectory]
        predicted_tool_calls = extract_tool_calls(full_trajectory)
        return sum(
            any(action.compare_with_tool_call(tool_call) for tool_call in predicted_tool_calls)
            for action in actions
        )

    @staticmethod
    def _normalize_observation_signature(content: str) -> str:
        return content.strip()

    def _record_step_label(self, label: int, reason: str | None = None) -> None:
        self.last_step_label = int(label)
        self.last_step_reason = reason
        self.step_label_history.append(int(label))

    def register_invalid_action(self, reason: str = "invalid_action_format") -> None:
        self.invalid_action_count += 1
        self._record_step_label(-1, reason)

    def register_answer(self, answer_text: str | None = None) -> None:
        del answer_text
        self._record_step_label(0, "answer")

    def _update_proxy_state(self, *, progress: bool, hard_bad: bool, reason: str | None, validate: bool) -> bool:
        if reason is not None:
            self.proxy_reason_history.append(reason)

        if progress:
            self.hard_bad_streak = 0
            self.soft_no_progress_streak = 0
            return False

        self.soft_no_progress_streak += 1
        if hard_bad:
            self.hard_bad_streak += 1
            self.cut = 1.0
        else:
            self.hard_bad_streak = 0
            self.stalling_counts += 1

        if validate or self.tolerate_num <= 0:
            return False

        if self.hard_bad_streak >= self.hard_tolerate_num:
            self.must_stop = 1.0
            self.cut = 1.0
            self.hard_truncation = 1
            return True

        if self.soft_no_progress_streak >= self.tolerate_num:
            self.must_stop = 1.0
            self.cut = 1.0
            self.soft_truncation = 1
            return True

        return False

    def compute_feedback(self, action_text: str, validate: bool = False):
        if not validate and self.must_stop == 1.0:
            self._record_step_label(0, "already_stopped")
            return None

        self.query_counts += 1
        repeated_action = False
        if action_text in self.all_queries:
            repeated_action = True
            self.repeat_counts += 1
            self._record_step_label(-1, "repeat_action")
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason="repeat_action",
                validate=validate,
            )
            if should_stop:
                return None
        self.all_queries.append(action_text)

        try:
            message = parse_action_string(action_text, requestor="assistant")
        except Exception as exc:
            self.invalid_action_count += 1
            self._record_step_label(-1, "invalid_tool_call")
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason="invalid_tool_call",
                validate=validate,
            )
            if should_stop:
                return None
            return f"Invalid tool call. Error: {exc}"

        if not isinstance(message, AssistantMessage) or not message.is_tool_call() or len(message.tool_calls) != 1:
            self.invalid_action_count += 1
            self._record_step_label(-1, "invalid_action_format")
            should_stop = self._update_proxy_state(
                progress=False,
                hard_bad=True,
                reason="invalid_action_format",
                validate=validate,
            )
            if should_stop:
                return None
            return (
                "Invalid action. Inside <interact> you must output exactly one tool call, "
                "for example create_task(user_id='user_1', title='Important Meeting')."
            )

        tool_call = message.tool_calls[0]
        if tool_call.name == "done":
            self.must_stop = 1.0
            self._record_step_label(0, "done")
            return None

        tool_type = self._get_tool_type(tool_call.name)
        state_before = self._get_env_state_signature()
        self.trajectory.append(message)
        response = self.env.get_response(tool_call)
        self.trajectory.append(response)
        state_after = self._get_env_state_signature()
        self.last_agent_db_hash, self.last_user_db_hash = state_after

        hard_bad = False
        progress = False
        reason = None

        if response.error:
            self.tool_error_count += 1
            hard_bad = True
            reason = "tool_error"
        else:
            matched_action_count = self._matched_action_count()
            action_progress = matched_action_count > self.prev_matched_action_count
            self.prev_matched_action_count = max(self.prev_matched_action_count, matched_action_count)

            state_changed = state_after != state_before
            if tool_type == ToolType.WRITE and not state_changed:
                self.no_state_change_write_count += 1
                hard_bad = True
                reason = "write_no_state_change"

            obs_signature = self._normalize_observation_signature(response.content)
            new_information = bool(obs_signature) and obs_signature not in self.seen_observation_signatures
            if obs_signature:
                self.seen_observation_signatures.add(obs_signature)

            if self.strict_progress_label:
                # Strict mode uses a cleaner ternary split:
                # progress: expected action coverage increases
                # neutral: all READs, or a WRITE that changes state but does not increase action coverage
                # bad: obvious invalid/error cases above, plus successful non-READ steps that neither
                #      advance expected actions nor change state
                progress = action_progress
                if not progress and reason is None:
                    if tool_type == ToolType.READ:
                        reason = "read_new_information" if new_information else "read_no_new_information"
                    elif state_changed:
                        reason = "state_changed_without_action_progress"
                    else:
                        hard_bad = True
                        reason = "successful_nonread_no_progress"
            else:
                progress = action_progress or (tool_type == ToolType.WRITE and state_changed)
                progress = progress or (tool_type == ToolType.READ and new_information)
                if not progress and reason is None:
                    reason = "no_progress"

        step_label = 1 if progress else -1 if (hard_bad or repeated_action) else 0
        self._record_step_label(step_label, reason)

        should_stop = self._update_proxy_state(
            progress=progress,
            hard_bad=hard_bad,
            reason=reason,
            validate=validate,
        )
        if should_stop:
            return None

        if self.query_counts >= self.max_counts:
            self.must_stop = 1.0
            return None

        tool_feedback = response.content
        suffix = "Now continue with the next tool call, or finish with <answer>done()</answer>."
        return f"Tool result: {tool_feedback}\n{suffix}"

    def finalize(self, official: bool = False):
        del official
        if self.reward_info is not None:
            return self.reward_info
        full_trajectory = [*self.initial_history, *self.trajectory]
        self.reward_info = evaluate_task(
            task=self.task,
            full_trajectory=full_trajectory,
            environment_constructor=self.env_constructor,
            reward_mode=self.reward_mode,
            solo_mode=True,
            env_kwargs=self.env_kwargs,
        )
        self.final_reward = float(self.reward_info.reward)
        return self.reward_info

    def format_trajectory(self) -> str:
        full_trajectory = [*self.initial_history, *self.trajectory]
        lines = []
        for item in full_trajectory:
            role = getattr(item, "role", "unknown")
            if role == "assistant" and getattr(item, "tool_calls", None):
                for tool_call in item.tool_calls:
                    args = ", ".join(
                        f"{key}={json.dumps(value, ensure_ascii=False)}"
                        for key, value in tool_call.arguments.items()
                    )
                    lines.append(f"[assistant] {tool_call.name}({args})")
            elif role == "tool":
                prefix = "[tool][error]" if getattr(item, "error", False) else "[tool]"
                lines.append(f"{prefix} {item.content}")
            else:
                content = getattr(item, "content", None)
                if content:
                    lines.append(f"[{role}] {content}")
        return "\n".join(lines)
