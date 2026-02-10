import sys, os
import time
import math
import pandas as pd
from openai import OpenAI

# Add parent directory to path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Internal Module Imports
try:
    from llm.llm_interface import query_llm, build_prompt
    from gating.token_gater import (
        gate, 
        gate_with_entropy
    )
    from utils.text_utils import count_tokens
except ImportError as e:
    print(f"Error: Missing local modules. {e}")
    sys.exit(1)

# Attempt to import pynvml for GPU memory measurement
try:
    import pynvml
except ImportError:
    print("Warning: pynvml library not found. GPU memory measurement disabled.")
    pynvml = None

# ==============================================================================
# 1. API and System Measurement Functions
# ==============================================================================

def query_token_optimiser(prompt_text, gating_mode=None):
    """Run the full Token Optimiser (gating + LLM) for evaluation."""
    start_time = time.perf_counter()

    # Logic for different gating strategies
    if gating_mode == "Entropy based gating":
        gated_prompt = gate_with_entropy(prompt_text)
    elif gating_mode == "original_gating":
        context = gate(prompt_text)
        gated_prompt = build_prompt(context, prompt_text)
    else:
        gated_prompt = prompt_text # Baseline

    response_text = query_llm(gated_prompt)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    prompt_tokens = count_tokens(gated_prompt)
    completion_tokens = count_tokens(response_text)
    tps = completion_tokens / total_time if total_time > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_sec": total_time,
        "tokens_per_second": tps,
        "response_text": response_text
    }

def get_gpu_memory_utilization(device_id: int = 0):
    if pynvml is None: return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"used_mib": mem_info.used / (1024 ** 2)}
    except Exception:
        return None

# ==============================================================================
# 2. Cost Calculation Functions (Paper-based)
# ==============================================================================

def calculate_inference_cost(n_in, c_in, n_out, c_out):
    return (n_in * c_in) + (n_out * c_out)

def calculate_cost_of_pass(total_cost, success_rate):
    if success_rate <= 0:
        return math.inf
    return total_cost / success_rate

# ==============================================================================
# 3. Main Evaluation Orchestrator
# ==============================================================================

def evaluate_model(model_config, client, query):
    print("\n" + "="*80)
    print(f"Starting Evaluation for Model: {model_config['name']}")
    print("="*80)
    
    gating_modes = [None, "original_gating", "Entropy based gating"]
    all_results = []

    for mode in gating_modes:
        mode_label = mode or 'baseline'
        print(f"\nRunning with gating mode: {mode_label}...")
        
        result = query_token_optimiser(query, gating_mode=mode)
        if result:
            result['gating_mode'] = mode_label
            # Calculate hypothetical cost using paper formula
            cost = calculate_inference_cost(
                result['prompt_tokens'], model_config['hypothetical_cost_per_input_token'],
                result['completion_tokens'], model_config['hypothetical_cost_per_output_token']
            )
            result['hypothetical_cost'] = cost
            # Success Rate (Manual or fixed for benchmarking)
            result['success_rate'] = 1.0 
            result['cost_of_pass'] = calculate_cost_of_pass(cost, result['success_rate'])
            
            all_results.append(result)
            print(f"    -> Generated in {result['total_time_sec']:.2f}s ({result['tokens_per_second']:.2f} TPS)")
        else:
            print(f"    -> Mode {mode_label} failed.")

    return pd.DataFrame(all_results)

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

def run_evaluation(query):
    MODEL_1_CONFIG = {
        "name": "liquid/lfm2-1.2b",
        "hypothetical_cost_per_input_token": 0.20 / 1_000_000,
        "hypothetical_cost_per_output_token": 1.00 / 1_000_000
    }
    
    # Optional: Add second model here
    models_to_evaluate = [MODEL_1_CONFIG]
    final_dfs = []
    
    if pynvml:
        pynvml.nvmlInit()
    
    try:
        for config in models_to_evaluate:
            input(f"\nLoad '{config['name']}' in LM Studio and press Enter...")
            
            mem_before = get_gpu_memory_utilization()
            
            report_df = evaluate_model(config, None, query)
            
            mem_after = get_gpu_memory_utilization()
            vram_footprint = (mem_after['used_mib'] - mem_before['used_mib']) if mem_before and mem_after else 0
            
            if not report_df.empty:
                report_df['model_name'] = config['name']
                report_df['vram_mib'] = vram_footprint
                final_dfs.append(report_df)

    finally:
        if pynvml:
            pynvml.nvmlShutdown()

    if final_dfs:
        full_report = pd.concat(final_dfs)
        print("\n" + "="*80)
        print("Final Comparison Report")
        print("="*80)
        print(full_report[['model_name', 'gating_mode', 'prompt_tokens', 'tokens_per_second', 'hypothetical_cost']].round(4).to_string(index=False))

if __name__ == "__main__":
    default_query = "What are the top 3 benefits of adopting a microservices architecture?"
    run_evaluation(default_query)