#!/usr/bin/env python3
"""
Compute all metrics (perplexity, BERTScore, BLEU) with multi-GPU support for perplexity
"""
import json
import argparse
import logging
import torch
import evaluate
import numpy as np
from bert_score import score
from pathlib import Path
import tempfile
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(jsonl_file, rank=0, world_size=1):
    """Load and optionally shard data for multi-GPU"""
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    if world_size > 1:
        chunk_size = len(data) // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank < world_size - 1 else len(data)
        data = data[start:end]
        logger.info(f"Rank {rank}/{world_size}: Processing {len(data)} samples ({start}-{end})")
    
    return data


def compute_perplexity_single_gpu(model_path, texts, batch_size, rank, output_file):
    """Compute perplexity on a single GPU"""
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        logger.warning(f"Rank {rank}: No valid texts")
        return
    
    logger.info(f"Rank {rank}: Computing perplexity for {len(texts)} texts...")
    
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(
        predictions=texts,
        model_id=model_path,
        batch_size=batch_size,
        add_start_token=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    output = {
        'perplexities': results['perplexities'],
        'num_samples': len(texts)
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f)
    
    logger.info(f"Rank {rank}: Saved results to {output_file}")


def merge_perplexity_results(temp_dir, num_gpus):
    """Merge perplexity results from all GPUs"""
    all_perplexities = []
    total_samples = 0
    
    for rank in range(num_gpus):
        rank_file = Path(temp_dir) / f"ppl_rank{rank}.json"
        if not rank_file.exists():
            logger.error(f"Missing results from rank {rank}")
            continue
        
        with open(rank_file, 'r') as f:
            result = json.load(f)
            all_perplexities.extend(result['perplexities'])
            total_samples += result['num_samples']
    
    if not all_perplexities:
        raise ValueError("No perplexity results found!")
    
    mean_ppl = float(np.mean(all_perplexities))
    logger.info(f"Merged perplexity from {total_samples} samples: {mean_ppl:.2f}")
    
    return {'mean_perplexity': mean_ppl, 'perplexities': all_perplexities}


def compute_bertscore(predictions, references, batch_size=64):
    """Compute BERTScore"""
    logger.info("Computing BERTScore...")
    P, R, F1 = score(predictions, references, lang="en", verbose=True, batch_size=batch_size)
    return {
        "precision": {"mean": P.mean().item(), "std": P.std().item()},
        "recall": {"mean": R.mean().item(), "std": R.std().item()},
        "f1": {"mean": F1.mean().item(), "std": F1.std().item()}
    }


def compute_bleu(predictions, references):
    """Compute BLEU"""
    logger.info("Computing BLEU...")
    sacrebleu = evaluate.load('sacrebleu')
    refs = [[ref] for ref in references]
    results = sacrebleu.compute(predictions=predictions, references=refs)
    return {"bleu": results['score'], "precisions": results['precisions']}


def print_summary(results):
    """Print summary of all metrics"""
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    if 'perplexity' in results:
        print(f"\nPerplexity: {results['perplexity']['mean_perplexity']:.2f}")
    if 'bertscore' in results:
        bs = results['bertscore']
        print(f"\nBERTScore F1: {bs['f1']['mean']:.4f} ± {bs['f1']['std']:.4f}")
    if 'sacrebleu' in results:
        print(f"BLEU: {results['sacrebleu']['bleu']:.2f}")
    if 'metadata' in results:
        print(f"\nSamples: {results['metadata']['num_samples']}")
    print("="*60 + "\n")


def run_multi_gpu_perplexity(predictions_file, model_path, num_gpus, batch_size):
    """Launch perplexity computation across multiple GPUs"""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temp directory: {temp_dir}")
    
    # Launch processes
    processes = []
    for rank in range(num_gpus):
        cmd = [
            sys.executable, __file__,
            '--mode', 'worker',
            '--predictions_file', predictions_file,
            '--model_path', model_path,
            '--batch_size', str(batch_size),
            '--rank', str(rank),
            '--world_size', str(num_gpus),
            '--output_file', f"{temp_dir}/ppl_rank{rank}.json"
        ]
        env = {'CUDA_VISIBLE_DEVICES': str(rank)}
        p = subprocess.Popen(cmd, env={**subprocess.os.environ, **env})
        processes.append(p)
    
    # Wait for all
    failed = False
    for rank, p in enumerate(processes):
        p.wait()
        if p.returncode != 0:
            logger.error(f"Rank {rank} failed with code {p.returncode}")
            failed = True
    
    if failed:
        raise RuntimeError("Perplexity computation failed")
    
    # Merge results
    perplexity_results = merge_perplexity_results(temp_dir, num_gpus)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return perplexity_results


def main():
    parser = argparse.ArgumentParser(description="Compute all metrics with multi-GPU support")
    parser.add_argument('--mode', choices=['main', 'worker'], default='main')
    parser.add_argument('--predictions_file', required=True)
    parser.add_argument('--model_path', default=None, help='Model path for perplexity (optional)')
    parser.add_argument('--output_file', default='metrics_results.json')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    
    args = parser.parse_args()
    
    if args.mode == 'worker':
        # Worker mode: compute perplexity for one GPU
        # CUDA_VISIBLE_DEVICES is already set, so use device 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        data = load_data(args.predictions_file, args.rank, args.world_size)
        references = [d['reference'] for d in data]
        compute_perplexity_single_gpu(args.model_path, references, args.batch_size, 
                                     args.rank, args.output_file)
    else:
        # Main mode: orchestrate everything
        logger.info("="*60)
        logger.info("COMPUTING ALL METRICS")
        logger.info("="*60)
        
        # Load full data
        data = load_data(args.predictions_file)
        predictions = [d['prediction'] for d in data]
        references = [d['reference'] for d in data]
        
        results = {'metadata': {'num_samples': len(data)}}
        
        # Step 1: Perplexity (multi-GPU) - optional
        if args.model_path:
            logger.info(f"[1/3] Computing perplexity on {args.num_gpus} GPUs...")
            results['perplexity'] = run_multi_gpu_perplexity(
                args.predictions_file, args.model_path, args.num_gpus, args.batch_size
            )
        else:
            logger.info("[1/3] Skipping perplexity (no model_path provided)")
        
        # Step 2: BERTScore (single GPU)
        logger.info("[2/3] Computing BERTScore...")
        try:
            results['bertscore'] = compute_bertscore(predictions, references)
        except Exception as e:
            logger.warning(f"BERTScore failed: {e}")
        
        # Step 3: BLEU (CPU)
        logger.info("[3/3] Computing BLEU...")
        try:
            results['sacrebleu'] = compute_bleu(predictions, references)
        except Exception as e:
            logger.warning(f"BLEU failed: {e}")
        
        # Save and display
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {args.output_file}")
        print_summary(results)


if __name__ == "__main__":
    main()
