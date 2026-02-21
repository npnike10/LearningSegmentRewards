#!/usr/bin/env python3

import optuna
import subprocess
import json
import os
from pathlib import Path


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e-4, log=True)
    beta = trial.suggest_float("beta", 0.01, 0.5, log=True)
    train_batch_size = trial.suggest_categorical(
        "train_batch_size", [64, 128, 256, 512]
    )
    max_epochs = trial.suggest_int("max_epochs", 1, 3)

    # Create unique checkpoint path for this trial
    trial_id = trial.number
    save_path = f"./checkpoint/qwen3-4B-dpo-rat1-optuna-trial-{trial_id}"
    tb_path = f"./checkpoint/qwen3-4B-dpo-rat1-optuna-trial-{trial_id}/"

    # Build command
    cmd = [
        "deepspeed",
        "--module",
        "openrlhf.cli.train_dpo",
        "--save_path",
        save_path,
        "--save_steps",
        "-1",
        "--use_tensorboard",
        tb_path,
        "--logging_steps",
        "1",
        "--eval_steps",
        "5",
        "--train_batch_size",
        str(train_batch_size),
        "--micro_train_batch_size",
        "8",
        "--pretrain",
        "./checkpoint/qwen3-4B-sft-rat2",
        "--bf16",
        "--max_epochs",
        str(max_epochs),
        "--max_len",
        "4096",
        "--zero_stage",
        "2",
        "--learning_rate",
        str(learning_rate),
        "--beta",
        str(beta),
        "--max_samples",
        "10000",
        "--dataset",
        "./data_processing/preferences/rationales_preference_train.parquet",
        "--eval_dataset",
        "./data_processing/preferences/rationales_preference_val.parquet",
        "--apply_chat_template",
        "--chosen_key",
        "chosen",
        "--rejected_key",
        "rejected",
        "--attn_implementation",
        "flash_attention_2",
        "--load_checkpoint",
        "--packing_samples",
        "--gradient_checkpointing",
        "--adam_offload",
    ]

    # Set environment variable
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    try:
        print(
            f"Starting Trial {trial_id}: lr={learning_rate:.2e}, beta={beta:.3f}, batch={train_batch_size}, epochs={max_epochs}"
        )

        # Run training
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=6000, env=env
        )

        if result.returncode != 0:
            print(f"Trial {trial_id} failed: {result.stderr}")
            raise optuna.TrialPruned()

        # Extract validation loss from logs
        print(f"Trial {trial_id}: Attempting to extract validation loss from {tb_path}")
        try:
            val_loss = extract_validation_loss(tb_path)
            print(
                f"Trial {trial_id}: Successfully extracted validation loss: {val_loss}"
            )
        except Exception as e:
            print(f"Trial {trial_id}: Error in extract_validation_loss: {e}")
            raise

        # Clean up checkpoint to save space
        if os.path.exists(save_path):
            subprocess.run(["rm", "-rf", save_path])

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

    tags = ea.Tags()

    # DPO typically logs eval/loss
    if "eval/eval_loss" in tags.get("scalars", []):
        print("Found eval/eval_loss, extracting...")
        eval_losses = ea.Scalars("eval/eval_loss")
        return min([scalar.value for scalar in eval_losses])
    else:
        raise ValueError("No val loss found")


if __name__ == "__main__":
    # Create study
    study = optuna.create_study(direction="minimize", study_name="dpo_tuning")

    print("Starting DPO hyperparameter optimization...")

    # Optimize
    study.optimize(objective, n_trials=15)

    # Print results
    print("\nOptimization completed!")
    if study.trials:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best parameters
        with open("best_dpo_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

        print("\nBest parameters saved to best_dpo_params.json")
    else:
        print("No trials completed successfully.")
