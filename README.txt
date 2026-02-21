# RLHF 
## Repository Structure
This tree structure highlights the key folders in this repository. Folders containing auxiliary code are not highlight for brevity. This readme file is the main entry point for instructions to run all components of this repository.
```
├── OpenRLHF/                    # Main RLHF training framework
│   ├── openrlhf/               # Core RLHF library modules
│   ├── data_processing/        # Scripts for creating finetuning datasets
│   ├── evaluation/             # Model evaluation and metrics
│   ├── checkpoint/             # Model checkpoints and training curves
│   ├── examples/               # Training scripts and examples
├── hyperparameter_tuning/      # Optuna-based hyperparameter optimization
```

## Installation
To install this repository for finetuning models, please clone this repository and follow the steps below:

```
# cd to OpenRLHF dir
# create docker image from dockerfile
docker build -t openrlhf-0:latest .
# create docker container
docker run --runtime=nvidia -it --name=openrlhf-ct --shm-size="10g" --cap-add=SYS_ADMIN -v $PWD:/OpenRLHF openrlhf-0:latest bash
# inside the docker container, change working dir to OpenRLHF repo dir. For example, if this installation was done in an EC2 instance and the docker container was created from the dir home/ec2-user, then use the line below, else modify as needed
cd ../OpenRLHF/OpenRLHF
# install required packages
pip install -e .
```
Further, please create a virtual environment to be used to run data processing and evaluation scripts for finetuned models:
```
python3 -m venv eval-env
source eval-env/bin/activate
pip install torch transformers datasets evaluate sacrebleu bert-score tqdm vllm accelerate scikit-learn jsonlines openpyxl pandas language_tool_python
```

## Model Finetuning
Ensure the working dir is `OpenRLHF` before running commands in this section. Please refer to the [OpenRLHF documentation](https://openrlhf.readthedocs.io/) as well as the file OpenRLHF-Original-Repo-README.md for more detailed information on finetuning models using OpenRLHF. 

The argument values in run command should be adjusted based on the machine used to run experiments, as per available compute and memory.

Before running commands in this section, please ensure to deactivate `eval-env` virtual environment.

To visualize training curves from tensorboard files of a run, please update the file_path and figure save path in checkpoint/plot_logs.py, then run the following command
```
python3 checkpoint/plot_logs.py
```

Note that unit-level training files are based on an older version of OpenRLHF, as compared to the other training files.
 
### SFT
Run the following command
```
deepspeed --module openrlhf.cli.train_sft \
   --max_len 4096\
   --dataset data_processing/sft/rationales_sft_train.parquet \
   --eval_dataset data_processing/sft/rationales_sft_val.parquet \
   --input_key messages \
   --apply_chat_template \
   --train_batch_size 64\
   --micro_train_batch_size 8\
   --max_samples 5000 \
   --pretrain Qwen/Qwen3-4B \
   --save_path ./checkpoint/qwen3-4B-sft-rat2 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 5 \
   --use_tensorboard ./checkpoint/qwen3-4B-sft-rat2/ \
   --zero_stage 2 \
   --max_epochs 3\
   --packing_samples \
   --bf16 \
   --learning_rate 4.68e-5 \
   --gradient_checkpointing \
   --adam_offload \
``` 

### DPO
Run the following command
```
VLLM_USE_FLASHINFER_SAMPLER=0 deepspeed --module openrlhf.cli.train_dpo --save_path ./checkpoint/qwen3-4B-sft-dpo-rat2 \
   --save_steps -1 \
   --use_tensorboard ./checkpoint/qwen3-4B-sft-dpo-rat2/ \
   --logging_steps 1 \
   --eval_steps 5\
   --train_batch_size 240\
   --micro_train_batch_size 3\
   --pretrain ./checkpoint/qwen3-4B-sft-rat2 \
   --bf16 \
   --max_epochs 1\
   --max_len 4096\
   --zero_stage 3\
   --learning_rate 5e-7 \
   --beta 0.1 \
   --max_samples 10000\
   --dataset ./data_processing/preferences/rationales_preference_train.parquet \
   --eval_dataset ./data_processing/preferences/rationales_preference_val.parquet \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --attn_implementation flash_attention_2 \
   --load_checkpoint \
   --packing_samples \
   --gradient_checkpointing \
   --adam_offload \
```

### Reward Model
Run the following command
```
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/qwen3-4B-sft-rm-rat2\
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps 5 \
   --train_batch_size 512\
   --micro_train_batch_size 32 \
   --pretrain ./checkpoint/qwen3-4B-sft-rat2 \
   --bf16 \
   --max_samples 10000 \
   --max_epochs 3\
   --max_len 4096 \
   --zero_stage 1 \
   --learning_rate 4e-5 \
   --dataset ./data_processing/preferences/rationales_preference_train.parquet \
   --eval_dataset ./data_processing/preferences/rationales_preference_val.parquet \
   --use_tensorboard ./checkpoint/qwen3-4B-sft-rm-rat2/ \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --packing_samples \
   --gradient_checkpointing \
   --value_head_prefix score
```

### PPO
Run the following command for PPO
```
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2\
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.75 \
   --packing_samples \
   --vllm_sync_backend gloo \
   --vllm_enable_sleep \
   --enforce_eager \
   --pretrain ./checkpoint/qwen3-4B-sft-rat2 \
   --reward_pretrain ./checkpoint/qwen3-4B-sft-rm-rat2 \
   --save_path ./checkpoint/qwen3-4B-sft-ppo-rat2/final \
   --ckpt_path ./checkpoint/qwen3-4B-sft-ppo-rat2/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size 16\
   --train_batch_size 512\
   --micro_rollout_batch_size 32\
   --rollout_batch_size 256\
   --n_samples_per_prompt 2\
   --enable_prefix_caching \
   --max_epochs 2\
   --prompt_max_len 4096\
   --max_samples 10000 \
   --generate_max_len 2048\
   --zero_stage 3\
   --bf16 \
   --actor_learning_rate 6.57e-6 \
   --critic_learning_rate 2.34e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ./data_processing/prompts/rationales_prompts_train.parquet \
   --input_key context_messages \
   --apply_chat_template \
   --use_tensorboard ./checkpoint/qwen3-4B-sft-ppo-rat2 \
   --normalize_reward \
   --adam_offload \
   --deepspeed_enable_sleep \
   --gradient_checkpointing \
```
To run variants like GRPO, RLOO etc., please refer to OpenRLHF_Original_Repo_README.md.

### DRCA
The unit extraction and reward files used for this algorithm are designed for Qwen3-4B runs with thinking turned off. For Qwen3-4B with thinking on, or a different model with different tokenizer, ensure to manually modify and test the relevant parts of code for correctness. For example, currently think tokens are removed, but they will need to be kept if thinking is turned on. 

Run the following command
```
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

python3 -m openrlhf.cli.train_drca_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --vllm_num_engines 2\
   --vllm_tensor_parallel_size 1 \
   --vllm_gpu_memory_utilization 0.75 \
   --packing_samples \
   --vllm_sync_backend gloo \
   --vllm_enable_sleep \
   --enforce_eager \
   --pretrain ./checkpoint/qwen3-4B-sft-rat2 \
   --reward_pretrain ./checkpoint/qwen3-4B-sft-rm-rat2 \
   --save_path ./checkpoint/qwen3-4B-sft-drca-cf-rat2/final \
   --ckpt_path ./checkpoint/qwen3-4B-sft-drca-cf-rat2/ckpt \
   --save_hf_ckpt \
   --save_steps -1\
   --logging_steps 1 \
   --micro_train_batch_size 16\
   --train_batch_size 512\
   --micro_rollout_batch_size 32\
   --rollout_batch_size 256\
   --n_samples_per_prompt 2\
   --enable_prefix_caching \
   --max_epochs 1\
   --prompt_max_len 4096\
   --max_samples 10000\
   --generate_max_len 2048\
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 6.57e-6 \
   --critic_learning_rate 2.34e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ./data_processing/prompts/rationales_prompts_train.parquet \
   --input_key context_messages \
   --apply_chat_template \
   --use_tensorboard ./checkpoint/qwen3-4B-sft-drca-cf-rat2/ \
   --gradient_checkpointing \
   --deepspeed_enable_sleep \
   --adam_offload \
   --normalize_reward \
   --max_units_per_sequence 10 \
   --drca_beta 0.8 \
   --unit_extraction "regex" \
```
Unit extraction can be 'regex' for step label based units, 'entropy' for entropy-based units.

### Unit-Level Reward Model
For more detailed information on unit-level reward model training and PPO, please refer to DenseRewardRLHF_PPO_README.md. This code builds on top of the DenseRewardRLHF-PPO repository by adding credit distribution entropy regularization term in segment reward model training loss.

Run the following command to train the unit-level reward model
```
bash examples/scripts/train_seg_rm.sh train_seg_rm qwen3 rationales 3 peak 1.5 avg 0.02
```
Above command runs 3 epochs of training, with entropy threshold for unit extraction as 1.5 and entropy regularization coefficient in training loss as 0.02. To switch from MaxEnt training loss to original DenseRewardRLHF paper's training loss, set the entropyregularization coefficient to zero.

#### Linear Function Fitting
The following command will create a folder 'fit_data' at `checkpoint/qwen3-4B-sft-denserm-rat2/final_model/fit_data` and generate json files in the folder. 
```
bash examples/scripts/run_fit_function.sh ../checkpoint/qwen3-4B-sft-denserm-rat2/final_model peak 1.5 avg
```

Before running the below command, ensure that input file path in line 75 in the script below correctly navigates to the newly generated json file for peak segment position rewards.
```
python3 useful_code/plot_logfit_function_sklearn.py
```
Above run will print Huber log fit params for mean and standard deviation in terminal output. For example, it may show 
```
Mean Huber Log Fit: y = ABC1log(x) + ABC2      
Mean R² value: 0.1278
Std Huber Log Fit: y = ABC3log(x) + ABC4
Std R² value: 0.0789
```
Before training with unit-level PPO, please create a `params.json` file located at ``checkpoint/qwen3-4B-sft-denserm-rat2/final_model/fit_data/` with the following contents (values are from above example, modify them with actual run values)
```
{
  "qwen3": {
    "rationales": {
      "peak": {
        "avg": {
          "1.5": [ABC1, ABC3, ABC2, ABC4]
        }
      }
    }
  }
}
```
1.5 represents entropy threshold, ensure it matches the value in training. The list elements correspond to coefficient of log(x) in mean huber log fit, coefficient of log(x) in std huber log fit, constant term in mean huber log fit, and constant term in std huber log fit, respectively. 

### Unit-Level PPO
Run the following command
```
bash examples/scripts/train_ppo_segment.sh ppo_segment_rm_training qwen3 rationales 1 peak 1.5 avg segment_normalization 0.5 rationales 0.1
```
To train model using standard PPO with outcome-level reward instead of unit-level rewards, use the following command
```
bash examples/scripts/train_ppo_segment.sh ppo_segment_rm_training qwen3 rationales 1 peak 1000 avg segment_last_avg 0.5 rationales
```

### Hyperparameter tuning
To tune hyperparameters, run the following command
```
python3 ../hyperparameter_tuning/optuna_<model>_tuning.py
```
where <model> should be replaced with 'sft', 'rm', 'dpo' or 'ppo' as appropriate.

The file being run can be modified to change the hyperparameters being optimized, the range of values considered or the optimization algorithm used. Please refer to Optuna documentation for more detailed information.

Best hyperparameters are printed at end of run and also saved in file `best_<model>_params.json` located at dir from where the tuning command was executed.

## Evaluation
Files here are designed for Qwen3-4B runs with thinking turned off. For Qwen3-4B with thinking on, or a different model with different tokenizer, ensure to manually modify and test the relevant parts of code for correctness. For example, currently think tokens are removed, but they will need to be kept if thinking is turned on. 

Before running commands in this section, please ensure to activate the `eval-env` virtual environment. Also ensure the working dir is `cd src/sharingan_internship_hub/eagle/ Automated_Procedure_Generation_Niket_niketnp@2025/OpenRLHF`.

### Batch inference
To run inference for finetuned model on test data, run the following command with <name> replaced by appropriate name
```
# first argument is model path, second is test dataset path, third is path for storing predictions from inference

evaluation/run_multigpu_inference.sh \
    ./checkpoint/<name>/final \ 
    ./data_processing/sft/rationales_sft_test.parquet \
    ./evaluation/generated_data/<name>.jsonl \
    8 \
    "" \
    2048 \
```

### Automated Metrics
To compute automated metrics (SacreBLEU, BERTScore, Perplexity), run the following command with <name> replaced by appropriate name
```
python3 evaluation/automated_metrics/compute_all_metrics.py \
  --predictions_file evaluation/generated_data/<name>.jsonl \
  --output_file evaluation/results/<name>.json \
  --num_gpus 8 \
  --batch_size 3
```

### LLM-as-a-Judge Metrics
To compute LLM-as-a-Judge metrics, run the following command with <name> replaced by appropriate name, and <metric> replaced by one of 'factual_accuracy', 'logical_soundness' or 'efficiency'.
```
python3 evaluation/llm-as-a-judge/<metric>_judge.py \
  --input evaluation/generated_data/<name>.jsonl \
  --output evaluation/results/<name>.json \
  --workers 50 \
  --max-samples 5000 \
```

To aggregate the obtained evaluation metric over test dataset, run the following command
```
python3 evaluation/llm-as-a-judge/aggregate_<metric>.py \
  --input evaluation/results/<name>.json \
```

### Reward Metric
This metric computes the average of normalized reward obtained for each prediction in given model predictions passed to given reward model.

Run the following command with <name> replaced by appropriate name
```
python3 evaluation/reward_metric.py --model_path ./checkpoint/<name> \
--data_path evaluation/generated_data/<name>.jsonl \
--output_path evaluation/results/<name>.jsonl \
``` 

### Zero-shot Claude 3.7 Sonnet Baseline
Run the following command
```
python3 evaluation/generate_baseline_rationales.py \
--input ./data_processing/sft/rationales_sft_test.parquet \
```
Output predictions will be located at `evaluation/generated_data`.