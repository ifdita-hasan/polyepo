#!/bin/bash

#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --time=96:00:00
#SBATCH --mem=564G
#SBATCH --output=%A.out
#SBATCH --error=%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --nodelist=iris-hgx-1,iris-hgx-2
#SBATCH --job-name="drgrpo"

# ============ Environment Setup ============
source /path/to/miniconda3/bin/activate
conda activate /path/to/envs/clean_verl/
cd /path/to/polyepo

unset ROCR_VISIBLE_DEVICES
export HF_HOME="/path/to/big_cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/path/to/big_cache/huggingface"
export RAY_TMPDIR=/tmp/ray
export PYTHONUNBUFFERED=1


export RAY_heartbeat_timeout_milliseconds=60000
export RAY_gcs_node_manager_max_heartbeat_misses=120

# ============ Ray Cluster Setup ============
ray stop --force

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export ip_head

# Start Ray head node
echo "Starting Ray HEAD on $head_node"
ray start --head \
    --node-ip-address="$head_node_ip" \
    --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus 4

# Start Ray worker nodes (if nodes > 1)
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" --export=ALL bash -c "
        source /path/to/miniconda3/bin/activate
        conda activate /path/to/envs/clean_verl/
        export RAY_TMPDIR=/tmp/ray
        ray start --address \"$ip_head\" \
            --num-cpus ${SLURM_CPUS_PER_TASK} \
            --num-gpus 4
    " &
    sleep 5
done

sleep 10
ray status

# ============ Hyperparameters & Paths ============
# RL with ground truth hyperparams
TRAIN_DATASET_PATH=/path/to/polyepo/data/polaris/train.parquet
TEST_DATASET_PATH="['/path/to/polyepo/data/math500/test.parquet','/path/to/polyepo/data/aime25/test.parquet','/path/to/polyepo/data/beyondaime/test.parquet','/path/to/polyepo/data/minerva/test.parquet']"

# PolyEPO specific settings from your original script
ALGORITHM=grpo
MODEL_PATH=Qwen/Qwen3-4B-Base
PROJECT_NAME=verl_polyppo
EXPERIMENT_NAME=drgrpo

PPO_EPOCHS=1
FULL_BATCH_SIZE=128
PPO_MINI_BATCH_SIZE=64
MICRO_BATCH_SIZE=1

ROLLOUT_N=8
REWARD_MANAGER='multi_thread'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ALGORITHM \
    data.train_files=$TRAIN_DATASET_PATH \
    data.val_files=$TEST_DATASET_PATH \
    data.train_batch_size=$FULL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.grad_clip=0.3 \
    actor_rollout_ref.actor.ppo_epochs=$PPO_EPOCHS \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.reward_manager=$REWARD_MANAGER \
    +polyppo_clustering.cluster_prompt_file=/path/to/polyepo/verl/trainer/ppo/cluster_prompts.py \
    +polyppo_clustering.cluster_prompt_fn='math_cluster_fn' \
    trainer.logger='["console", "wandb"]' \
    trainer.val_before_train=True \
    trainer.val_on_last_step=True \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=1 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=5 \
    +ray_kwargs.ray_init.num_cpus=16 \
    ray_init.ray_dir=$RAY_TMPDIR 2>&1 | tee verl_demo.log
