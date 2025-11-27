#!/usr/bin/env python3
"""
Quick diagnostic script to check sequence length distribution
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Load dataset
data_path = 'ORLM/data/OR-Instruct-Data-3K-Gurobipy.jsonl'
dataset = load_dataset('json', data_files=data_path, split='train')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', trust_remote_code=True)

# Format and tokenize
def format_and_tokenize(example):
    prompt = (example.get('prompt') or '').strip()
    completion = (example.get('completion') or '').strip()
    text = f"{prompt}\n{completion}"
    tokens = tokenizer(text, truncation=False, add_special_tokens=True)
    return {'length': len(tokens['input_ids'])}

print("Analyzing sequence lengths...")
dataset = dataset.map(format_and_tokenize)

lengths = dataset['length']
lengths_array = np.array(lengths)

print(f"\nSequence Length Statistics:")
print(f"  Total examples: {len(lengths)}")
print(f"  Mean length: {lengths_array.mean():.1f}")
print(f"  Median length: {np.median(lengths_array):.1f}")
print(f"  Min length: {lengths_array.min()}")
print(f"  Max length: {lengths_array.max()}")
print(f"  25th percentile: {np.percentile(lengths_array, 25):.1f}")
print(f"  75th percentile: {np.percentile(lengths_array, 75):.1f}")
print(f"  95th percentile: {np.percentile(lengths_array, 95):.1f}")
print(f"  99th percentile: {np.percentile(lengths_array, 99):.1f}")

# Count how many exceed common limits
print(f"\nExamples exceeding common limits:")
print(f"  > 512 tokens: {(lengths_array > 512).sum()} ({100 * (lengths_array > 512).sum() / len(lengths):.1f}%)")
print(f"  > 1024 tokens: {(lengths_array > 1024).sum()} ({100 * (lengths_array > 1024).sum() / len(lengths):.1f}%)")
print(f"  > 2048 tokens: {(lengths_array > 2048).sum()} ({100 * (lengths_array > 2048).sum() / len(lengths):.1f}%)")
print(f"  > 4096 tokens: {(lengths_array > 4096).sum()} ({100 * (lengths_array > 4096).sum() / len(lengths):.1f}%)")

# Show distribution
print(f"\nLength distribution (bins):")
bins = [0, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, float('inf')]
bin_labels = ['0-256', '256-512', '512-768', '768-1024', '1024-1536',
              '1536-2048', '2048-3072', '3072-4096', '4096-8192', '8192+']
for i in range(len(bins)-1):
    count = ((lengths_array >= bins[i]) & (lengths_array < bins[i+1])).sum()
    pct = 100 * count / len(lengths)
    print(f"  {bin_labels[i]:12s}: {count:4d} examples ({pct:5.1f}%)")

print("\nRecommendation:")
optimal_length = int(np.percentile(lengths_array, 95))
print(f"  Set max_length to {optimal_length} to cover 95% of examples")
print(f"  This will speed up training by reducing padding overhead")
