#!/usr/bin/env python3

import optuna
import subprocess
import json
import os
import tempfile
from pathlib import Path

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-8, 1e-4, log=True)
    train_batch_size = trial.suggest_categorical('train_batch_size', [64, 128, 256, 512])
    max_epochs = trial.suggest_int('max_epochs', 1, 3)
    
    # Create unique checkpoint path for this trial
    trial_id = trial.number
    checkpoint_path = f"./checkpoint/hparam_tuning/rationales/qwen3-4B-sft-rat1-optuna-trial-{trial_id}"
    
    # Build command
    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_sft",
        "--max_len", "4096",
        "--dataset", "data_processing/sft/rationales_sft_train.parquet",
        "--eval_dataset", "data_processing/sft/rationales_sft_val.parquet",
        "--input_key", "messages",
        "--apply_chat_template",
        "--train_batch_size", str(train_batch_size),
        "--micro_train_batch_size", "8",
        "--max_samples", "5000",
        "--pretrain", "Qwen/Qwen3-4B",
        "--save_path", checkpoint_path,
        "--save_steps", "-1",
        "--logging_steps", "1",
        "--eval_steps", "2",
        "--use_tensorboard", checkpoint_path,
        "--zero_stage", "2",
        "--max_epochs", str(max_epochs),
        "--packing_samples",
        "--bf16",
        "--learning_rate", str(learning_rate),
        "--gradient_checkpointing",
        "--adam_offload"
    ]
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"Trial {trial_id} failed: {result.stderr}")
            raise optuna.TrialPruned()
        
        # Extract validation loss from logs
        val_loss = extract_validation_loss(checkpoint_path)
        
        # Clean up checkpoint to save space
        if os.path.exists(checkpoint_path):
            subprocess.run(["rm", "-rf", checkpoint_path])
        
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
    
    # Simple approach: parse the latest log file for eval loss
    # In practice, you might want to use tensorboard's event reader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    ea = EventAccumulator(str(log_files[-1]))
    ea.Reload()
    
    if 'eval/eval gpt_loss' in ea.Tags()['scalars']:
        eval_losses = ea.Scalars('eval/eval gpt_loss')
        return min([scalar.value for scalar in eval_losses])
    else:
        # Fallback: return a high loss if eval loss not found
        raise ValueError("No val loss found")

if __name__ == "__main__":
    # Create study
    study = optuna.create_study(
        direction='minimize',
        study_name='sft_tuning'
    )
    
    print("Starting SFT hyperparameter optimization...")
    
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
        with open('best_sft_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        print("\nBest parameters saved to best_sft_params.json")
    else:
        print("No trials completed successfully.")