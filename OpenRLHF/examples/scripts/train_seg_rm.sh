exp_prefix=$1
model_type=$2
dataset_type=$3
epoch=$4
segment_method=$5
entropy_threshold=$6
agg_func=$7

# cache_dir="/checkpoint/reward_model/${exp_prefix}/${exp_prefix}_${model_type}_${dataset_type}_${epoch}epoch_segmethod-${segment_method}_entropy-${entropy_threshold}_aggfunc-${agg_func}"
cache_dir="../checkpoint/qwen3-4B-sft-denserm-rat2-et${entropy_threshold}-reg"
data_dir="${cache_dir}/datasets"

cd openrlhf
ACCELERATE_LOG_LEVEL=info

if [[ "$model_type" == *"phi3-instruct"* ]]; then
    model="microsoft/Phi-3-mini-4k-instruct"
    micro_train_batch_size=8
elif [[ "$model_type" == *"rlhflow_llama_3_sft_8b_v2"* ]]; then
    model="RLHFlow/LLaMA3-SFT-v2"
    micro_train_batch_size=4
elif [[ "$model_type" == *"qwen3"* ]]; then
    model="../checkpoint/qwen3-4B-sft-rat2"
    micro_train_batch_size=2
elif [[ "$model_type" == *"meta_llama_3_1_instruct_8b"* ]]; then
    model="meta-llama/Llama-3.1-8B-Instruct"
    micro_train_batch_size=4
else
    model="EleutherAI/pythia-70m"
    micro_train_batch_size=4
fi

echo "reward model init from:" ${model}

torchrun --nproc_per_node=1 --nproc_per_node=8 cli/train_denserm.py \
    --pretrain ${model} \
    --output_root_dir ${cache_dir} \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps 5 \
    --micro_train_batch_size ${micro_train_batch_size} \
    --train_batch_size 512 \
    --max_epochs ${epoch} \
    --max_len 4096 \
    --zero_stage 3 \
    --bf16 \
    --learning_rate 4e-5 \
    --flash_attn \
    --gradient_checkpointing \
    --dataset ../data_processing/preferences/rationales_preference_train.parquet \
    --eval_dataset ../data_processing/preferences/rationales_preference_val.parquet \
    --use_tensorboard  ../checkpoint/qwen3-4B-sft-denserm-rat2-et${entropy_threshold}-reg/ \
    --apply_chat_template \
    --chosen_key chosen \
    --rejected_key rejected \
    --packing_samples \
    --exp_prefix ${exp_prefix} \
    --entropy_threshold ${entropy_threshold} \
    --agg_func ${agg_func} \
    --segment_method ${segment_method} \
    --load_checkpoint \
    --value_head_prefix score \
    --adam_offload \
    --entropy_regularizer \