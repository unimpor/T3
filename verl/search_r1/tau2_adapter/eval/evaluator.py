from search_r1.tau2_adapter.core.tasks import RewardType
from search_r1.tau2_adapter.eval.action import evaluate_actions
from search_r1.tau2_adapter.eval.env import evaluate_environment
from search_r1.tau2_adapter.eval.schema import RewardInfo


def evaluate_task(
    task,
    full_trajectory,
    environment_constructor,
    reward_mode: str = "action",
    solo_mode: bool = True,
    env_kwargs: dict | None = None,
    fractional: bool = False,
) -> RewardInfo:
    reward_mode = reward_mode.lower()
    if reward_mode not in {"action", "env", "action_env", "task_basis"}:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")

    action_reward = evaluate_actions(task, full_trajectory, fractional=fractional)
    env_reward = evaluate_environment(
        environment_constructor=environment_constructor,
        task=task,
        full_trajectory=full_trajectory,
        solo_mode=solo_mode,
        env_kwargs=env_kwargs,
        respect_reward_basis=reward_mode == "task_basis",
        fractional=fractional,
    )

    if reward_mode == "action":
        return action_reward
    if reward_mode == "env":
        return env_reward
    if reward_mode == "task_basis":
        reward_basis = {
            item.value if isinstance(item, RewardType) else str(item)
            for item in (task.evaluation_criteria.reward_basis if task.evaluation_criteria is not None else [])
        }
        unsupported_basis = reward_basis - {
            RewardType.DB.value,
            RewardType.ENV_ASSERTION.value,
            RewardType.ACTION.value,
            RewardType.COMMUNICATE.value,
        }
        if unsupported_basis:
            raise NotImplementedError(
                f"task_basis reward currently does not support reward types: {sorted(unsupported_basis)}"
            )

        reward = 1.0
        reward_breakdown = {}
        info = {
            "action": action_reward.info,
            "env": env_reward.info,
        }

        if fractional:
            atomic_scores: list[float] = []
            if RewardType.DB.value in reward_basis and env_reward.db_check is not None:
                atomic_scores.append(float(env_reward.db_check.db_reward))
                reward_breakdown["DB"] = float(env_reward.db_check.db_reward)
            if RewardType.ENV_ASSERTION.value in reward_basis:
                if env_reward.env_assertions:
                    env_scores = [float(check.reward) for check in env_reward.env_assertions]
                    atomic_scores.extend(env_scores)
                    reward_breakdown["ENV_ASSERTION"] = sum(env_scores) / len(env_scores)
                else:
                    reward_breakdown["ENV_ASSERTION"] = 1.0
            if RewardType.ACTION.value in reward_basis:
                if action_reward.action_checks:
                    action_scores = [float(check.action_reward) for check in action_reward.action_checks]
                    atomic_scores.extend(action_scores)
                    reward_breakdown["ACTION"] = sum(action_scores) / len(action_scores)
                else:
                    reward_breakdown["ACTION"] = 1.0
            if RewardType.COMMUNICATE.value in reward_basis:
                communicate_info = task.evaluation_criteria.communicate_info if task.evaluation_criteria is not None else None
                if communicate_info:
                    raise NotImplementedError(
                        "fractional task_basis reward with COMMUNICATE checks is not implemented in the VERL tau2 adapter yet."
                    )
                atomic_scores.append(1.0)
                reward_breakdown["COMMUNICATE"] = 1.0
                info["communicate"] = {"reward": 1.0, "checks": []}

            reward = sum(atomic_scores) / len(atomic_scores) if atomic_scores else 1.0
            return RewardInfo(
                reward=reward,
                action_checks=action_reward.action_checks,
                db_check=env_reward.db_check,
                env_assertions=env_reward.env_assertions,
                reward_breakdown=reward_breakdown,
                info=info,
            )

        if reward_basis & {RewardType.DB.value, RewardType.ENV_ASSERTION.value}:
            reward *= env_reward.reward
            reward_breakdown.update(env_reward.reward_breakdown)
        if RewardType.ACTION.value in reward_basis:
            reward *= action_reward.reward
            reward_breakdown.update(action_reward.reward_breakdown)
        if RewardType.COMMUNICATE.value in reward_basis:
            communicate_info = task.evaluation_criteria.communicate_info if task.evaluation_criteria is not None else None
            if communicate_info:
                raise NotImplementedError(
                    "task_basis reward with COMMUNICATE checks is not implemented in the VERL tau2 adapter yet."
                )
            reward_breakdown["COMMUNICATE"] = 1.0
            info["communicate"] = {"reward": 1.0, "checks": []}

        return RewardInfo(
            reward=reward,
            action_checks=action_reward.action_checks,
            db_check=env_reward.db_check,
            env_assertions=env_reward.env_assertions,
            reward_breakdown=reward_breakdown,
            info=info,
        )

    reward = action_reward.reward * env_reward.reward
    reward_breakdown = {}
    reward_breakdown.update(action_reward.reward_breakdown)
    reward_breakdown.update(env_reward.reward_breakdown)
    return RewardInfo(
        reward=reward,
        action_checks=action_reward.action_checks,
        db_check=env_reward.db_check,
        env_assertions=env_reward.env_assertions,
        reward_breakdown=reward_breakdown,
        info={
            "action": action_reward.info,
            "env": env_reward.info,
        },
    )
