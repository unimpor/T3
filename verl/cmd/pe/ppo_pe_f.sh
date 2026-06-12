#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="PE-F"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/${DATA_NAME}}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints}"
NUM_GPUS="${NUM_GPUS:-8}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_dtg_full_attr8_v2.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val_dtg_full_attr8_v2.parquet}"

AS_BONUS="${AS_BONUS:-true}"
BT_BONUS="${BT_BONUS:-true}"
BONUS_MODE="${BONUS_MODE:-minority_fixed}"
BONUS_SCALE="${BONUS_SCALE:-1.0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen7B-gated-turn12-attr8-dtg-as-bt-v2-continuous}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

project_name="${DATA_NAME}-Train"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${EXPERIMENT_NAME}"

mkdir -p "${default_local_dir}"

python3 -m verl.trainer.main_ppo \
    max_turns="${MAX_TURNS:-12}" \
    data_name="${DATA_NAME}" \
    use_interactions=false \
    early_cut=false \
    +trunc_strength="${TRUNC_STRENGTH:-0.5}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-512}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-512}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-6656}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
    data.max_start_length="${MAX_START_LENGTH:-2048}" \
    data.max_obs_length="${MAX_OBS_LENGTH:-512}" \
    algorithm.adv_estimator=gae \
    algorithm.as_bonus="${AS_BONUS}" \
    algorithm.bt_bonus="${BT_BONUS}" \
    algorithm.bonus_mode="${BONUS_MODE}" \
    algorithm.bonus_scale="${BONUS_SCALE}" \
    algorithm.as_cf=false \
    algorithm.bt_cf=false \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-128}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size="${REF_LOG_PROB_MICRO_BATCH_SIZE:-128}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    critic.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    critic.optim.lr="${CRITIC_LR:-1e-5}" \
    critic.model.use_remove_padding=true \
    critic.optim.lr_warmup_steps_ratio="${CRITIC_WARMUP_RATIO:-0.015}" \
    critic.model.path="${BASE_MODEL}" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    trainer.logger="${TRAINER_LOGGER:-[\"console\"]}" \
    trainer.val_only=false \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-true}" \
    trainer.default_hdfs_dir=null \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq="${SAVE_FREQ:-250}" \
    trainer.test_freq="${TEST_FREQ:-6}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-200}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-200}" \
    trainer.default_local_dir="${default_local_dir}" \
    2>&1 | tee "${default_local_dir}.log"
