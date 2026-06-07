#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="Tau2Bench"
DOMAIN_NAME="${DOMAIN_NAME:-telecom}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/${DATA_NAME}/${DOMAIN_NAME}}"
BASE_MODEL="${BASE_MODEL:-${PROJECT_ROOT}/models/Qwen/Qwen2.5-14B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints}"
NUM_GPUS="${NUM_GPUS:-8}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_full_minus_test_9009_standard_think_short.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val_full_minus_test_9009_standard_think_short.parquet}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

export TAU2_STANDARD_USER_API_KEY="${TAU2_STANDARD_USER_API_KEY:?Set TAU2_STANDARD_USER_API_KEY}"
export TAU2_STANDARD_USER_AZURE_ENDPOINT="${TAU2_STANDARD_USER_AZURE_ENDPOINT:?Set TAU2_STANDARD_USER_AZURE_ENDPOINT}"
export TAU2_STANDARD_USER_API_VERSION="${TAU2_STANDARD_USER_API_VERSION:-2024-02-01}"
export TAU2_STANDARD_USER_MODEL="${TAU2_STANDARD_USER_MODEL:-gpt-4.1-2025-04-14}"
export TAU2_STANDARD_USER_TEMPERATURE="${TAU2_STANDARD_USER_TEMPERATURE:-0.0}"
export TAU2_STANDARD_USER_SEED="${TAU2_STANDARD_USER_SEED:-300}"
export TAU2_STANDARD_USER_MAX_TOOL_HOPS="${TAU2_STANDARD_USER_MAX_TOOL_HOPS:-3}"
export TAU2_STANDARD_USER_MAX_TOKENS="${TAU2_STANDARD_USER_MAX_TOKENS:-256}"
export TAU2_STANDARD_USER_LOGID="${TAU2_STANDARD_USER_LOGID:-}"
export TAU2_STANDARD_USER_MAX_RETRIES="${TAU2_STANDARD_USER_MAX_RETRIES:-4}"
export TAU2_STANDARD_USER_RETRY_BASE_SECONDS="${TAU2_STANDARD_USER_RETRY_BASE_SECONDS:-1.0}"
export TAU2_STANDARD_USER_SIM_PARALLELISM="${TAU2_STANDARD_USER_SIM_PARALLELISM:-12}"
export TAU2_STANDARD_USER_DISCONNECT_FALLBACK="${TAU2_STANDARD_USER_DISCONNECT_FALLBACK:-true}"
export TAU2_STANDARD_USER_DISCONNECT_MESSAGE="${TAU2_STANDARD_USER_DISCONNECT_MESSAGE:-Sorry, I got disconnected for a moment. Could you repeat that?}"

COMPLETION_BONUS="${COMPLETION_BONUS:-0.0}"
EARLY_CUT="${EARLY_CUT:-false}"
T3_K="${T3_K:-3}"
TAU2_T3_PROGRESS_MODE="${TAU2_T3_PROGRESS_MODE:-new_family}"
TAU2_T3_MIN_TURN="${TAU2_T3_MIN_TURN:-6}"

project_name="${DATA_NAME}-${DOMAIN_NAME}-Train"
experiment_name="${EXPERIMENT_NAME:-standard-baseline-14B-ppo-vanilla}"
default_local_dir="${OUTPUT_ROOT}/${project_name}/${experiment_name}"

mkdir -p "${default_local_dir}"

python3 -m verl.trainer.main_ppo \
    data_name="${DATA_NAME}" \
    max_turns=22 \
    use_interactions=false \
    early_cut="${EARLY_CUT}" \
    +trunc_strength="${T3_K}" \
    +hard_tolerate_num=3 \
    +tau2_strict_progress_label=false \
    +tau2_completion_bonus="${COMPLETION_BONUS}" \
    +tau2_t3_progress_mode="${TAU2_T3_PROGRESS_MODE}" \
    +tau2_t3_min_turn="${TAU2_T3_MIN_TURN}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=20480 \
    data.max_response_length=768 \
    data.max_start_length=20480 \
    data.truncation=middle \
    data.max_obs_length=4096 \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=65536 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=65536 \
    \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=false \
    algorithm.use_arew_bonus=false \
    \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    critic.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=true \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path="${BASE_MODEL}" \
    critic.model.enable_gradient_checkpointing=true \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    \
    trainer.logger='["console", "wandb"]' \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=6 \
    trainer.total_epochs=200 \
    trainer.log_val_generations=128 \
    trainer.total_training_steps=150 \
    trainer.default_local_dir="${default_local_dir}" \
    2>&1 | tee "${default_local_dir}.log"
