exp_prefix=$1
model_type=$2
dataset_type=$3
num_episodes=$4
segment_method=$5
entropy_threshold=$6
agg_func=$7
ppo_reward_type=$8
value_clip=$9
reward_fit_dataset=${10}
lambda_reg=${11}

reward_mean=0
reward_std=1

if [ "$model_type" = "phi3-instruct" ]; then
    pretrain_model="microsoft/Phi-3-mini-4k-instruct"
    micro_train_batch_size=4
    micro_rollout_batch_size=16

    if [ "$entropy_threshold" = "1.75" ]; then
        rm_model_dir="yyqoni/Phi-3-mini-4k-instruct-segment-rm-700k"
    elif [ "$entropy_threshold" = "0" ]; then
        rm_model_dir="yyqoni/Phi-3-mini-4k-instruct-token-rm-700k"
    elif [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.51953125
        reward_std=2.15625
        rm_model_dir="yyqoni/Phi-3-mini-4k-instruct-bandit-rm-700k"
    fi
elif [ "$model_type" = "rlhflow_llama_3_sft_8b_v2" ]; then
    pretrain_model="RLHFlow/LLaMA3-SFT-v2"
    micro_train_batch_size=2
    micro_rollout_batch_size=4
    if [ "$entropy_threshold" = "2" ]; then
        rm_model_dir="yyqoni/rlhflow-llama-3-sft-8b-v2-segment-rm-700k"
    elif [ "$entropy_threshold" = "0" ]; then
        rm_model_dir="yyqoni/rlhflow-llama-3-sft-8b-v2-token-rm-700k"
    elif [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.81640625
        reward_std=2.953125
        rm_model_dir="yyqoni/rlhflow-llama-3-sft-8b-v2-bandit-rm-700k"
    fi
elif [ "$model_type" = "meta_llama_3_1_instruct_8b" ]; then
    pretrain_model="meta-llama/Llama-3.1-8B-Instruct"
    micro_train_batch_size=2
    micro_rollout_batch_size=4
    if [ "$entropy_threshold" = "2" ]; then
        rm_model_dir="yyqoni/meta-llama-3.1-instruct-8b-segment-rm-700k"
    elif [ "$entropy_threshold" = "0" ]; then
        rm_model_dir="yyqoni/meta-llama-3-1-instruct-8b-token-rm-700k"
    elif [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.828125
        reward_std=2.9375
        rm_model_dir="yyqoni/meta-llama-3-1-instruct-8b-bandit-rm-700k"
    fi
elif [ "$model_type" = "qwen3" ]; then
    pretrain_model="../checkpoint/qwen3-4B-sft-rat2"
    micro_train_batch_size=4
    micro_rollout_batch_size=8
    if [ "$lambda_reg" = "0" ]; then
        rm_model_dir="../checkpoint/qwen3-4B-sft-denserm-rat2-et${entropy_threshold}/final_model/"
    else
        rm_model_dir="../checkpoint/qwen3-4B-sft-denserm-rat2-et${entropy_threshold}-reg${lambda_reg}/final_model/"
    fi
    if [ "$entropy_threshold" = "1000" ]; then
        reward_mean=-0.667309
        reward_std=5.772278
    fi
else
    pretrain_model=EleutherAI/pythia-70m
    rm_model_dir=EleutherAI/pythia-70m
fi


policy_model_dir="../checkpoint/qwen3-4B-sft-denserm-ppo-rat2-et${entropy_threshold}-rep1"

echo "Pretrain Model: $pretrain_model"
echo "Reward Model Directory: $rm_model_dir"
echo "Policy Model Directory: $policy_model_dir"
echo "Reward Mean: $reward_mean"
echo "Reward Std: $reward_std"

cd openrlhf

torchrun --nproc_per_node=8 cli/train_denserm_ppo.py \
    --pretrain ${pretrain_model} \
    --reward_pretrain ${rm_model_dir} \
    --output_root_dir ${policy_model_dir} \
    --exp_prefix ${exp_prefix} \
    --save_steps -1 \
    --logging_steps 1 \
    --micro_train_batch_size ${micro_train_batch_size} \
    --train_batch_size 128 \
    --micro_rollout_batch_size ${micro_rollout_batch_size} \
    --rollout_batch_size 512 \
    --num_episodes ${num_episodes} \
    --prompt_max_len 4096 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 1.8e-5 \
    --critic_learning_rate 7.13e-6 \
    --init_kl_coef 0.01 \
    --prompt_data ../data_processing/prompts/rationales_prompts_train.parquet \
    --input_key context_messages \
    --apply_chat_template \
    --n_samples_per_prompt 2 \
    --max_samples 10000 \
    --actor_init_on_gpu \
    --flash_attn \
    --gradient_checkpointing \
    --entropy_threshold ${entropy_threshold} \
    --ppo_reward_type ${ppo_reward_type} \
    --reward_mean ${reward_mean} \
    --reward_std ${reward_std} \
    --value_clip ${value_clip} \
    --normalize_reward \
    --segment_method ${segment_method} \
    --agg_func ${agg_func} \
    --norm_params_path ${rm_model_dir}/../fit_data/params.json \
    --reward_fit_dataset ${reward_fit_dataset} \
    --adam_offload \
    --use_tensorboard ${policy_model_dir} \
