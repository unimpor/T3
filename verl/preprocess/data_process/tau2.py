import argparse
import os
import random

from datasets import Dataset

from search_r1.tau2_adapter.loader.tasks import load_tasks
from search_r1.tau2_adapter.loader.registry import get_env_constructor
from search_r1.tau2_adapter.prompts import build_solo_prompt, build_standard_prompt_messages
from search_r1.tau2_adapter.domains.mock.environment import get_tasks_split as mock_get_tasks_split
from search_r1.tau2_adapter.domains.telecom.environment import get_tasks_split as telecom_get_tasks_split


def get_task_splits(domain: str) -> dict[str, list[str]]:
    if domain == "mock":
        return mock_get_tasks_split()
    if domain == "telecom":
        return telecom_get_tasks_split()
    raise ValueError(f"Custom split construction is not supported for domain={domain}")


def main():
    parser = argparse.ArgumentParser(description="Convert Tau2Bench tasks into parquet files for training and evaluation.")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--domain", type=str, default="mock")
    parser.add_argument("--task_split", type=str, default="base")
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=2)
    parser.add_argument("--reward_mode", type=str, default="action")
    parser.add_argument("--policy_type", type=str, default="manual")
    parser.add_argument("--source_split", type=str, default=None)
    parser.add_argument("--exclude_splits", type=str, default="")
    parser.add_argument("--custom_split_name", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="solo", choices=["solo", "standard"])
    parser.add_argument("--enable_think", action="store_true")
    parser.add_argument("--think_mode", type=str, default="short")
    parser.add_argument("--train_output", type=str, default=None)
    parser.add_argument("--val_output", type=str, default=None)
    args = parser.parse_args()

    env_constructor = get_env_constructor(args.domain)
    env_kwargs = {}
    if args.domain == "telecom":
        env_kwargs["policy_type"] = args.policy_type

    def to_record(task, idx, split_name, controller_task_split=None, task_pool_name=None):
        solo_mode = args.mode == "solo"
        env = env_constructor(solo_mode=solo_mode, **env_kwargs)
        if solo_mode:
            prompt = [{"role": "user", "content": build_solo_prompt(
                task,
                env,
                enable_think=args.enable_think,
                think_mode=args.think_mode,
            )}]
        else:
            prompt = build_standard_prompt_messages(
                task,
                env,
                enable_think=args.enable_think,
                think_mode=args.think_mode,
            )
        reward_mode = args.reward_mode
        if args.mode == "standard":
            reward_mode = "task_basis"
        elif task.evaluation_criteria is not None and task.evaluation_criteria.env_assertions:
            reward_mode = "action_env"
        controller = {
            "domain": args.domain,
            "task_id": task.id,
            "task_split": controller_task_split if controller_task_split is not None else split_name,
            "mode": args.mode,
            "reward_mode": reward_mode,
        }
        if args.enable_think:
            controller["enable_think"] = True
            controller["think_mode"] = args.think_mode
        if task_pool_name is not None:
            controller["task_pool"] = task_pool_name
        if env_kwargs:
            controller["env_kwargs"] = env_kwargs
        return {
            # Keep a non-empty struct so HuggingFace Datasets can write parquet.
            # Tau2 reward is injected during rollout via rm_scores, so this field is metadata-only.
            "answer": {
                "task_id": task.id,
                "domain": args.domain,
                "mode": args.mode,
            },
            "data_source": f"tau2_{args.domain}",
            "prompt": prompt,
            "ability": "tool-use",
            "reward_model": {"style": "external"},
            "controller": controller,
            "extra_info": {
                "split": split_name,
                "index": idx,
            },
        }

    output_dir = os.path.join(args.local_dir, "Tau2Bench", args.domain)
    os.makedirs(output_dir, exist_ok=True)
    think_suffix = f"_think_{args.think_mode}" if args.enable_think else ""
    mode_suffix = f"_{args.mode}"

    def resolve_output_path(default_filename: str, override: str | None) -> str:
        return override if override is not None else os.path.join(output_dir, default_filename)

    if args.source_split is not None:
        if not 0.0 < args.train_ratio < 1.0:
            raise ValueError(f"--train_ratio must be between 0 and 1, got {args.train_ratio}")

        task_splits = get_task_splits(args.domain)
        if args.source_split not in task_splits:
            raise ValueError(
                f"Invalid --source_split={args.source_split}. Valid splits are: {sorted(task_splits.keys())}"
            )

        exclude_split_names = [name.strip() for name in args.exclude_splits.split(",") if name.strip()]
        invalid_excludes = [name for name in exclude_split_names if name not in task_splits]
        if invalid_excludes:
            raise ValueError(
                f"Invalid exclude split names: {invalid_excludes}. Valid splits are: {sorted(task_splits.keys())}"
            )

        source_ids = list(task_splits[args.source_split])
        excluded_ids = set()
        for split_name in exclude_split_names:
            excluded_ids.update(task_splits[split_name])
        candidate_ids = [task_id for task_id in source_ids if task_id not in excluded_ids]
        if len(candidate_ids) < 2:
            raise ValueError(
                f"Not enough tasks left after filtering: source_split={args.source_split}, "
                f"exclude_splits={exclude_split_names}, remaining={len(candidate_ids)}"
            )

        all_tasks = load_tasks(args.domain, task_split_name=None, solo_only=args.mode == "solo")
        task_by_id = {task.id: task for task in all_tasks}
        missing_ids = [task_id for task_id in candidate_ids if task_id not in task_by_id]
        if missing_ids:
            raise ValueError(
                f"Missing solo-compatible tasks for requested custom split: "
                f"{missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}"
            )

        selected_tasks = [task_by_id[task_id] for task_id in candidate_ids]
        rng = random.Random(args.seed)
        rng.shuffle(selected_tasks)

        train_size = int(len(selected_tasks) * args.train_ratio)
        train_size = min(max(train_size, 1), len(selected_tasks) - 1)
        train_tasks = selected_tasks[:train_size]
        val_tasks = selected_tasks[train_size:]

        split_label = args.custom_split_name
        if split_label is None:
            if exclude_split_names:
                split_label = f"{args.source_split}_minus_{'_'.join(exclude_split_names)}"
            else:
                split_label = args.source_split
        split_label = split_label.replace(",", "_").replace(" ", "_")

        train_records = [
            to_record(
                task,
                idx,
                split_name=f"{split_label}_train",
                controller_task_split=args.source_split,
                task_pool_name=split_label,
            )
            for idx, task in enumerate(train_tasks)
        ]
        val_records = [
            to_record(
                task,
                idx,
                split_name=f"{split_label}_val",
                controller_task_split=args.source_split,
                task_pool_name=split_label,
            )
            for idx, task in enumerate(val_tasks)
        ]
        ratio_tag = f"{int(args.train_ratio * 100):02d}{int((1 - args.train_ratio) * 100):02d}"
        train_path = resolve_output_path(
            f"train_{split_label}_{ratio_tag}{mode_suffix}{think_suffix}.parquet",
            args.train_output,
        )
        val_path = resolve_output_path(
            f"val_{split_label}_{ratio_tag}{mode_suffix}{think_suffix}.parquet",
            args.val_output,
        )
        print(
            f"Constructed custom split '{split_label}' from source_split={args.source_split}, "
            f"exclude_splits={exclude_split_names or '[]'}, total={len(selected_tasks)}, "
            f"train={len(train_tasks)}, val={len(val_tasks)}, seed={args.seed}"
        )
    elif args.train_split is not None or args.val_split is not None:
        if args.train_split is None or args.val_split is None:
            raise ValueError("Please provide both --train_split and --val_split, or neither.")
        train_tasks = load_tasks(args.domain, task_split_name=args.train_split, solo_only=args.mode == "solo")
        val_tasks = load_tasks(args.domain, task_split_name=args.val_split, solo_only=args.mode == "solo")
        if len(train_tasks) == 0 or len(val_tasks) == 0:
            raise ValueError(
                f"Empty split detected for domain={args.domain}: "
                f"train_split={args.train_split} ({len(train_tasks)}), "
                f"val_split={args.val_split} ({len(val_tasks)})"
            )
        train_records = [to_record(task, idx, args.train_split) for idx, task in enumerate(train_tasks)]
        val_records = [to_record(task, idx, args.val_split) for idx, task in enumerate(val_tasks)]
        train_path = resolve_output_path(f"train_{args.train_split}{mode_suffix}{think_suffix}.parquet", args.train_output)
        val_path = resolve_output_path(f"val_{args.val_split}{mode_suffix}{think_suffix}.parquet", args.val_output)
    else:
        tasks = load_tasks(args.domain, task_split_name=args.task_split, solo_only=args.mode == "solo")
        if len(tasks) < 2:
            raise ValueError(
                f"Not enough compatible tasks found for domain={args.domain}, split={args.task_split}, mode={args.mode}"
            )
        records = [to_record(task, idx, args.task_split) for idx, task in enumerate(tasks)]
        val_size = min(max(args.val_size, 1), len(records) - 1)
        train_records = records[:-val_size]
        val_records = records[-val_size:]
        train_path = resolve_output_path(f"train_{args.task_split}{mode_suffix}{think_suffix}.parquet", args.train_output)
        val_path = resolve_output_path(f"val_{args.task_split}{mode_suffix}{think_suffix}.parquet", args.val_output)

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Saved train dataset to {train_path} ({len(train_dataset)} samples)")
    print(f"Saved val dataset to {val_path} ({len(val_dataset)} samples)")


if __name__ == "__main__":
    main()
