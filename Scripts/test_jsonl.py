import json
import sys
import os
import re
import time
import pandas as pd
import numpy as np
from openai import OpenAI

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.llm_interface import query_llm
from evaluation.evaluator import query_token_optimiser, calculate_inference_cost, calculate_cost_of_pass

# Attempt to import pynvml
try:
    import pynvml
except ImportError:
    pynvml = None

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def translate_text(text):
    translation_prompt = (
        f"Translate the following Chinese text to English. "
        f"Output ONLY the English translation without any introductory or concluding remarks.\n\n"
        f"{text}"
    )
    try:
        # Using a distinct call to avoid mixing up metrics
        response = query_llm(translation_prompt)
        return response.strip()
    except Exception:
        return text

def process_jsonl(file_path, limit=100):
    print(f"Reading prompts from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print(f"Found {len(lines)} lines. Processing first {limit} items...\n")

    # Metrics storage
    all_results = []
    
    # Config for cost calc (Hypothetical)
    COST_IN = 0.20 / 1_000_000
    COST_OUT = 1.00 / 1_000_000
    
    # Gating modes to test: only Baseline and the Custom Gating (Original)
    GATING_MODES = [None, "original_gating"]

    count = 0
    for i, line in enumerate(lines):
        if count >= limit:
            break
        
        if not line.strip():
            continue
            
        try:
            data = json.loads(line)
            raw_query = data.get('text', '') or data.get('prompt', '') or data.get('title', '')
            
            if not raw_query:
                continue
            
            count += 1
            print(f"Processing Item {count}/{limit}...")
            
            # Translate
            final_query = raw_query
            if contains_chinese(raw_query):
                final_query = translate_text(raw_query[:1000])

            # Test each gating mode
            for mode in GATING_MODES:
                mode_name = mode if mode else "baseline (no gating)"
                try:
                    # Run evaluation
                    # Note: We are not measuring VRAM per query here to save time/complexity, 
                    # as it requires clearing cache/server restarts for precision.
                    result = query_token_optimiser(final_query, gating_mode=mode) 
                    
                    # Calculate cost
                    cost = calculate_inference_cost(
                        result['prompt_tokens'], COST_IN,
                        result['completion_tokens'], COST_OUT
                    )
                    
                    # Store
                    record = {
                        "item_id": i,
                        "gating_mode": mode_name,
                        "prompt_tokens": result['prompt_tokens'],
                        "completion_tokens": result['completion_tokens'],
                        "latency": result['total_time_sec'],
                        "tps": result['tokens_per_second'],
                        "cost": cost
                    }
                    all_results.append(record)
                    
                except Exception as e:
                    print(f"  Failed mode {mode_name}: {e}")

        except json.JSONDecodeError:
            continue
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    # Analysis
    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS (First {count} items)")
    print("="*80)

    # Group by gating mode
    summary = df.groupby('gating_mode').agg({
        'prompt_tokens': 'mean',
        'completion_tokens': 'mean',
        'latency': 'mean',
        'tps': 'mean',
        'cost': 'sum' # Total cost for the batch
    }).reset_index()

    # Rename for clarity
    summary = summary.rename(columns={
        'prompt_tokens': 'Avg Input Tok',
        'completion_tokens': 'Avg Output Tok',
        'latency': 'Avg Latency (s)',
        'tps': 'Avg TPS',
        'cost': 'Total Batch Cost ($)'
    })

    print(summary.to_string(index=False))
    
    print("\n" + "="*80)
    print("Detailed stats saved to 'evaluation_results_100.csv'")
    df.to_csv("evaluation_results_100.csv", index=False)

if __name__ == "__main__":
    jsonl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'zh_general.jsonl')
    process_jsonl(jsonl_path, limit=100)
