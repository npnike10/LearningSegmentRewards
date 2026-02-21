#!/usr/bin/env python3

import optuna
import subprocess
import json
import os
import tempfile
from pathlib import Path


def objective(trial):
    # Suggest hyperparameters
    actor_learning_rate = trial.suggest_float(
        "actor_learning_rate", 1e-8, 1e-4, log=True
    )
    critic_learning_rate = trial.suggest_float(
        "critic_learning_rate", 1e-8, 1e-4, log=True
    )
    train_batch_size = trial.suggest_categorical("train_batch_size", [128, 256, 512])
    rollout_batch_size = trial.suggest_categorical(
        "rollout_batch_size", [256, 512, 1024]
    )
    max_epochs = trial.suggest_int("max_epochs", 1, 3)

    # Create unique checkpoint path for this trial
    trial_id = trial.number
    save_path = f"checkpoint/qwen3-4B-ppo-rat1-optuna-trial-{trial_id}/final"
    ckpt_path = f"checkpoint/qwen3-4B-ppo-rat1-optuna-trial-{trial_id}/ckpt"
    tb_path = f"checkpoint/qwen3-4B-ppo-rat1-optuna-trial-{trial_id}"

    # Build command
    cmd = [
        "python3",
        "-m",
        "openrlhf.cli.train_ppo_ray",
        "--ref_num_nodes",
        "1",
        "--ref_num_gpus_per_node",
        "1",
        "--reward_num_nodes",
        "1",
        "--reward_num_gpus_per_node",
        "1",
        "--critic_num_nodes",
        "1",
        "--critic_num_gpus_per_node",
        "2",
        "--actor_num_nodes",
        "1",
        "--actor_num_gpus_per_node",
        "2",
        "--vllm_num_engines",
        "2",
        "--vllm_tensor_parallel_size",
        "1",
        "--vllm_gpu_memory_utilization",
        "0.75",
        "--packing_samples",
        "--vllm_sync_backend",
        "gloo",
        "--vllm_enable_sleep",
        "--enforce_eager",
        "--pretrain",
        "checkpoint/qwen3-4B-sft-rat2",
        "--reward_pretrain",
        "checkpoint/qwen3-4B-sft-rm-rat2",
        "--save_path",
        save_path,
        "--ckpt_path",
        ckpt_path,
        "--save_hf_ckpt",
        "--micro_train_batch_size",
        "16",
        "--train_batch_size",
        str(train_batch_size),
        "--micro_rollout_batch_size",
        "32",
        "--rollout_batch_size",
        str(rollout_batch_size),
        "--n_samples_per_prompt",
        "2",
        "--enable_prefix_caching",
        "--max_epochs",
        str(max_epochs),
        "--prompt_max_len",
        "4096",
        "--max_samples",
        "10000",
        "--generate_max_len",
        "2048",
        "--zero_stage",
        "3",
        "--bf16",
        "--actor_learning_rate",
        str(actor_learning_rate),
        "--critic_learning_rate",
        str(critic_learning_rate),
        "--init_kl_coef",
        "0.01",
        "--prompt_data",
        "data_processing/prompts/rationales_prompts_train.parquet",
        "--input_key",
        "context_messages",
        "--apply_chat_template",
        "--use_tensorboard",
        tb_path,
        "--normalize_reward",
        "--gradient_checkpointing",
        "--deepspeed_enable_sleep",
        "--adam_offload",
    ]

    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20000)

        if result.returncode != 0:
            print(f"Trial {trial_id} failed: {result.stderr}")
            raise optuna.TrialPruned()

        # Extract validation loss from logs
        val_loss = extract_validation_loss(tb_path)

        # Clean up checkpoint to save space
        if os.path.exists(f"checkpoint/qwen3-4B-ppo-rat1-optuna-trial-{trial_id}"):
            subprocess.run(
                ["rm", "-rf", f"checkpoint/qwen3-4B-ppo-rat1-optuna-trial-{trial_id}"]
            )

        return val_loss

    except subprocess.TimeoutExpired:
        print(f"Trial {trial_id} timed out")
        raise optuna.TrialPruned()
    except Exception as e:
        print(f"Trial {trial_id} error: {e}")
        raise optuna.TrialPruned()


def extract_validation_loss(checkpoint_path):
    """Extract the best validation loss from training logs"""
    log_files = list(Path(checkpoint_path).rglob("events.out.tfevents.*"))

    if not log_files:
        raise ValueError("No tensorboard logs found")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(log_files[-1]))
    ea.Reload()

    if "train/return" in ea.Tags()["scalars"]:
        eval_losses = ea.Scalars("train/return")
        return min([scalar.value for scalar in eval_losses])
    else:
        raise ValueError("No training return found")


if __name__ == "__main__":
    # Create study
    study = optuna.create_study(direction="maximize", study_name="ppo_tuning")

    print("Starting PPO hyperparameter optimization...")

    # Optimize
    study.optimize(objective, n_trials=20)

    # Print results
    print("\nOptimization completed!")
    if study.trials:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best parameters
        with open("best_ppo_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

        print("\nBest parameters saved to best_ppo_params.json")
    else:
        print("No trials completed successfully.")
