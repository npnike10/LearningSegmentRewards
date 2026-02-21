"""
Fast batch inference using OpenRLHF's vLLM integration
"""
import json
import argparse
import re
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(test_file: str, max_samples: int = None, gpu_id: int = 0, num_gpus: int = 1, is_IFEval_data: bool = False):
    """Load test data from parquet, JSONL file, or HuggingFace dataset"""
    import pandas as pd
    from pathlib import Path
    
    # Check if it's a HuggingFace dataset (no file extension or contains '/')
    if '/' in test_file and not Path(test_file).suffix:
        from datasets import load_dataset
        logger.info(f"Loading HuggingFace dataset: {test_file}")
        
        try:
            dataset = load_dataset(test_file, split='test')
            logger.info(f"Loaded 'test' split with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"No 'test' split available for dataset {test_file}")
            raise ValueError(f"Dataset {test_file} does not have a 'test' split") from e
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # Split data across GPUs
        if num_gpus > 1:
            chunk_size = len(dataset) // num_gpus
            start_idx = gpu_id * chunk_size
            end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else len(dataset)
            dataset = dataset.select(range(start_idx, end_idx))
            logger.info(f"GPU {gpu_id}: Processing samples {start_idx} to {end_idx}")
        
        if is_IFEval_data:
            user_prompts = [sample['prompt'] for sample in dataset]
            messages_list = [[{'role': 'user', 'content': p}] for p in user_prompts]
            references = [''] * len(user_prompts)
        else:
            key_name = 'messages' if 'messages' in dataset.column_names else 'chosen'
            messages_list = [sample[key_name] for sample in dataset]
            user_prompts = [msgs[0]['content'] for msgs in messages_list]
            references = [msgs[1]['content'] for msgs in messages_list]
    
    elif test_file.endswith('.parquet'):
        df = pd.read_parquet(test_file)
        if max_samples:
            df = df.head(max_samples)
        
        # Split data across GPUs
        if num_gpus > 1:
            chunk_size = len(df) // num_gpus
            start_idx = gpu_id * chunk_size
            end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else len(df)
            df = df.iloc[start_idx:end_idx]
            logger.info(f"GPU {gpu_id}: Processing samples {start_idx} to {end_idx}")
        
        if is_IFEval_data:
            user_prompts = df['prompt'].tolist()
            messages_list = [[{'role': 'user', 'content': p}] for p in user_prompts]
            references = [''] * len(user_prompts)
        else:
            key_name = 'messages' if 'messages' in df.columns else 'chosen'
            messages_list = df[key_name].tolist()
            user_prompts = [msg[0]['content'] for msg in messages_list]
            references = [msg[1]['content'] for msg in messages_list]
    else:
        # JSONL fallback
        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        
        # Split data across GPUs
        if num_gpus > 1:
            chunk_size = len(data) // num_gpus
            start_idx = gpu_id * chunk_size
            end_idx = start_idx + chunk_size if gpu_id < num_gpus - 1 else len(data)
            data = data[start_idx:end_idx]
            logger.info(f"GPU {gpu_id}: Processing samples {start_idx} to {end_idx}")
        
        if is_IFEval_data:
            user_prompts = [d['prompt'] for d in data]
            messages_list = [[{'role': 'user', 'content': p}] for p in user_prompts]
            references = [''] * len(user_prompts)
        else:
            key_name = 'messages' if 'messages' in data[0] else 'chosen'
            messages_list = [d[key_name] for d in data]
            user_prompts = [msg[0]['content'] for msg in messages_list]
            references = [msg[1]['content'] for msg in messages_list]
    
    logger.info(f"Loaded {len(user_prompts)} test samples")
    return messages_list, user_prompts, references

def batch_inference_vllm(model_path: str, messages_list: list, **generation_kwargs):
    """
    Fast batch inference using vLLM with chat template
    """
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        
        gpu_memory_util = generation_kwargs.get('gpu_memory_utilization', 0.9)
        
        logger.info(f"Using vLLM, GPU memory util: {gpu_memory_util}")
        
        # Load tokenizer for chat template
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        
        # Initialize vLLM
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_util
        )
        
        # Apply chat template to convert messages to prompts
        prompts = [tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True) for msgs in messages_list]
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=generation_kwargs.get('temperature', 0.7),
            top_p=generation_kwargs.get('top_p', 0.8),
            min_p=generation_kwargs.get('min_p', 0),
            top_k=generation_kwargs.get('top_k', 20),
            max_tokens=generation_kwargs.get('max_new_tokens', 512),
            skip_special_tokens=False
        )
        
        # Generate
        logger.info(f"Generating predictions for {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params)
        
        predictions = []
        for output in outputs:
            output_ids = output.outputs[0].token_ids
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
                clean_ids = output_ids[index:]
                text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip("\n")  # Skip here instead
            except ValueError:
                text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            predictions.append(text.strip())
        
        return predictions
        
    except Exception as e:
        logger.warning(f"vLLM failed: {e}, falling back to transformers")
        return batch_inference_transformers(model_path, messages_list, **generation_kwargs)

def batch_inference_transformers(model_path: str, messages_list: list, **generation_kwargs):
    """
    Batch inference using transformers (fallback) with chat template
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info("Using transformers for inference")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    # Apply chat template to convert messages to prompts
    prompts = [tokenizer.apply_chat_template([msgs[0]], tokenize=False, add_generation_prompt=True) for msgs in messages_list]
    
    predictions = []
    batch_size = generation_kwargs.get('batch_size', 8)
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=generation_kwargs.get('max_length', 4096)
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=generation_kwargs.get('max_new_tokens', 512),
                temperature=generation_kwargs.get('temperature', 0.7),
                top_p=generation_kwargs.get('top_p', 0.8),
                top_k=generation_kwargs.get('top_k', 20),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            input_length = encoded.input_ids[j].shape[0]
            generated_tokens = output[input_length:]
            
            # Remove think tokens using token ID approach
            output_ids = generated_tokens.tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
                clean_ids = output_ids[index:]
                generated_text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip("\n")
            except ValueError:
                generated_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            predictions.append(generated_text.strip())
    
    return predictions

def save_predictions(user_prompts: list, predictions: list, references: list, output_file: str, is_IFEval_data: bool = False):
    """Save predictions to JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt, pred, ref in zip(user_prompts, predictions, references):
            if is_IFEval_data:
                json.dump({
                    'prompt': prompt,
                    'response': pred
                }, f, ensure_ascii=False)
            else:
                json.dump({
                    'prompt': prompt,
                    'prediction': pred,
                    'reference': ref
                }, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Fast batch inference using OpenRLHF model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--test_file", required=True, help="Path to test parquet/JSONL file")
    parser.add_argument("--output_file", default="predictions.jsonl", help="Output file for predictions")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum number of samples")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID for multi-GPU inference")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (for transformers)")
    parser.add_argument("--max_length", type=int, default=4096, help="Max input length (for transformers)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.01, help="Top-p sampling")
    parser.add_argument("--min_p", type=float, default=0, help="Min-p sampling")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization for vLLM (default: 0.9)")
    parser.add_argument("--use_vllm", action="store_true", help="Force use vLLM")
    parser.add_argument("--is_IFEval_data", action="store_true", help="Load IFEval format data (prompt column only)")
    
    args = parser.parse_args()
    
    # Load test data
    messages_list, user_prompts, references = load_test_data(args.test_file, args.max_samples, args.gpu_id, args.num_gpus, args.is_IFEval_data)
    
    # Generation parameters
    generation_kwargs = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_p": args.min_p,
        "top_k": args.top_k,
        "presence_penalty": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization
    }
    
    # Run inference
    if args.use_vllm:
        predictions = batch_inference_vllm(args.model_path, messages_list, **generation_kwargs)
    else:
        try:
            predictions = batch_inference_vllm(args.model_path, messages_list, **generation_kwargs)
        except:
            predictions = batch_inference_transformers(args.model_path, messages_list, **generation_kwargs)
    
    # Save predictions
    save_predictions(user_prompts, predictions, references, args.output_file, args.is_IFEval_data)
    
    # Print sample
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {user_prompts[i][:150]}...")
        print(f"Prediction: {predictions[i][:150]}...")
        print(f"Reference: {references[i][:150]}...")

if __name__ == "__main__":
    main()
