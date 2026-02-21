import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from tqdm import tqdm
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer

def compute_rewards(model_path, data_path, output_path=None):
    model = get_llm_for_sequence_regression(model_path, "reward", normalize_reward=True, init_value_head=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = get_tokenizer(model_path, model, "left")
    
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_json(data_path, lines=True)
    
    rewards = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        messages = [{'role': 'user', 'content': row['prompt']}, {'role': 'assistant', 'content': row['prediction']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        attn = torch.ones_like(inputs)
        with torch.no_grad():
            reward = model(inputs, attention_mask=attn).item()
        rewards.append(reward)
    
    df['reward'] = rewards
    
    if output_path:
        df.to_json(output_path, orient='records', lines=True) if output_path.endswith('.json') else df.to_parquet(output_path, index=False)
    
    print(f"Mean Reward: {sum(rewards)/len(rewards):.4f}")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()
    
    compute_rewards(args.model_path, args.data_path, args.output_path)
