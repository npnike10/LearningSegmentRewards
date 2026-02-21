import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer

model_path = "./checkpoint/qwen3-4B-sft-rm-rat1"
data_path = "./data_processing/preferences/rationales_preference_test.parquet"

model = get_llm_for_sequence_regression(model_path, "reward", normalize_reward=True, init_value_head=False)
model.eval()

print(f"Model normalization: {model.normalize_reward}, mean={model.mean.item():.4f}, std={model.std.item():.4f}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = get_tokenizer(model_path, model, "left")

df = pd.read_parquet(data_path)

chosen_rewards = []
rejected_rewards = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    chosen = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    rejected = tokenizer.apply_chat_template(row["rejected"], tokenize=False)

    chosen_inputs = tokenizer(chosen, return_tensors="pt").input_ids.to(device)
    rejected_inputs = tokenizer(rejected, return_tensors="pt").input_ids.to(device)

    chosen_attn = torch.ones_like(chosen_inputs)
    rejected_attn = torch.ones_like(rejected_inputs)

    with torch.no_grad():
        chosen_reward = model(chosen_inputs, attention_mask=chosen_attn).item()
        rejected_reward = model(rejected_inputs, attention_mask=rejected_attn).item()

    chosen_rewards.append(chosen_reward)
    rejected_rewards.append(rejected_reward)

chosen_rewards = np.array(chosen_rewards)
rejected_rewards = np.array(rejected_rewards)
margins = chosen_rewards - rejected_rewards

accuracy = (margins > 0).mean()

print(f"\n{'='*50}")
print("REWARD MODEL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Correct: {(margins > 0).sum()}/{len(margins)}")
print(f"\nReward Margin Statistics:")
print(f"  Mean: {margins.mean():.4f}")
print(f"  Median: {np.median(margins):.4f}")
print(f"  Std: {margins.std():.4f}")
print(f"\nChosen Reward Statistics:")
print(f"  Mean: {chosen_rewards.mean():.4f}")
print(f"  Std: {chosen_rewards.std():.4f}")
print(f"\nRejected Reward Statistics:")
print(f"  Mean: {rejected_rewards.mean():.4f}")
print(f"  Std: {rejected_rewards.std():.4f}")
