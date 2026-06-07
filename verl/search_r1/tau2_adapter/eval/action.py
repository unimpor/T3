from search_r1.tau2_adapter.core.message import AssistantMessage, ToolCall, UserMessage
from search_r1.tau2_adapter.core.tasks import Action, Task
from search_r1.tau2_adapter.eval.schema import ActionCheck, RewardInfo


def extract_tool_calls(full_trajectory) -> list[ToolCall]:
    tool_calls = []
    for message in full_trajectory:
        if isinstance(message, (AssistantMessage, UserMessage)) and message.is_tool_call():
            tool_calls.extend(message.tool_calls)
    return tool_calls


def evaluate_actions(task: Task, full_trajectory, fractional: bool = False) -> RewardInfo:
    if task.evaluation_criteria is None or not task.evaluation_criteria.actions:
        return RewardInfo(reward=1.0, reward_breakdown={"ACTION": 1.0}, info={"note": "No actions to evaluate"})

    predicted_tool_calls = extract_tool_calls(full_trajectory)
    action_checks = []
    for action in task.evaluation_criteria.actions:
        found = any(action.compare_with_tool_call(tool_call) for tool_call in predicted_tool_calls)
        action_checks.append(
            ActionCheck(
                action=action,
                action_match=found,
                action_reward=1.0 if found else 0.0,
            )
        )
    if fractional:
        reward = sum(check.action_reward for check in action_checks) / len(action_checks)
    else:
        reward = 1.0 if all(check.action_match for check in action_checks) else 0.0
    return RewardInfo(
        reward=reward,
        action_checks=action_checks,
        reward_breakdown={"ACTION": reward},
    )
