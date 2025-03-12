
HDFS_HOME=/home/ubuntu/chongli
WANDB_HANDLE=TODO
RUN_NAME=Qwen2.5-Math-7B_ppo_from_base_math_lv35

python3 /home/ubuntu/chongli/simpleRL-reason/train/openrlhf/cli/train_ppo_ray_env.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
    --save_path /home/ubuntu/chongli/checkpoint/test-rl-bash \
    --micro_train_batch_size 2 \
    --train_batch_size 32 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 128 \
    --temperature 0.6 \
    --n_samples_per_prompt 8 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data  /home/ubuntu/joey/swe-tests/bashbenchtasks.json \
    --input_key task_description \
    --apply_chat_template \
    --normalize_reward \
    --env_file bash_bench_env \
    --env_class BashBenchEnv \
    --flash_attn \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 4 \
    --load_checkpoint \
    --use_wandb $WANDB_HANDLE \
    --wandb_run_name bash_simple_check \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000