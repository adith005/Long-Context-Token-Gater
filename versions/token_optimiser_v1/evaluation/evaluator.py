#
# LLM Evaluation Script for LM Studio
#
# This script evaluates models hosted on an LM Studio local server, measuring
# performance, memory usage, and cost-efficiency metrics.
#
# The cost calculation functions are based on the paper "Efficient Agents:
# Building Effective Agents While Reducing Cost"[cite: 3].
#
# The code for API interaction and GPU memory measurement is a standard
# implementation and is NOT from the provided paper.
#
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llm.llm_interface import query_llm, build_prompt
from gating.token_gater import gate

import time
import math
from openai import OpenAI

# Attempt to import pynvml for GPU memory measurement
try:
    import pynvml
except ImportError:
    print("Warning: pynvml library not found. GPU memory measurement will be disabled.")
    print("To enable memory measurement, run: pip install pynvml")
    pynvml = None

# ==============================================================================
# 1. API and System Measurement Functions
# ==============================================================================

def query_token_optimiser(prompt_text):
    """Run the full Token Optimiser (gating + LLM) for evaluation."""
    start_time = time.perf_counter()
    context = gate(prompt_text)
    prompt = build_prompt(context, prompt_text)
    response_text = query_llm(prompt)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    # Token counting
    from utils.text_utils import count_tokens
    prompt_tokens = count_tokens(prompt)
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
    """Checks the GPU memory utilization for a specific NVIDIA GPU."""
    if pynvml is None: return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"used_mib": mem_info.used / (1024 ** 2)}
    except pynvml.NVMLError:
        return None

# ==============================================================================
# 2. Cost Calculation Functions (from the provided paper)
# ==============================================================================

def calculate_inference_cost(num_input_tokens: int, cost_per_input_token: float,
                             num_output_tokens: int, cost_per_output_token: float) -> float:
    """
    Calculates the cost of an inference attempt, C_m(p).
    This function implements the formula:
    C_m(p) = n_in(m,p) * c_in(m) + n_out(m,p) * c_out(m).
    """
    return (num_input_tokens * cost_per_input_token) + (num_output_tokens * cost_per_output_token)

def calculate_cost_of_pass(total_inference_cost: float, success_rate: float) -> float:
    """
    Calculates the cost-of-pass metric, v(m,p).
    This function implements the formula v(m,p) = C_m(p) / R_m(p).
    """
    if success_rate == 0:
        # As noted in the paper, cost-of-pass is infinity if accuracy is zero[cite: 114].
        return math.inf
    return total_inference_cost / success_rate

# ==============================================================================
# 3. Main Evaluation Orchestrator
# ==============================================================================

def evaluate_model(model_config, client):
    """Runs the full evaluation suite for a single model."""
    print("\n" + "="*80)
    print(f"Starting Evaluation for Model: {model_config['name']}")
    print("="*80)
    
    # User interaction to load the model
    input(f"Please load the '{model_config['name']}' model in LM Studio and start the server.\n"
          "Press Enter when you are ready to begin the evaluation...")

    # --- Memory Measurement (Before) ---
    mem_before = get_gpu_memory_utilization()
    if mem_before:
        print(f"Initial VRAM Usage: {mem_before['used_mib']:.2f} MiB")

    # --- Performance Evaluation ---
    sample_prompts = [
        "What are the top 3 benefits of adopting a microservices architecture?",
        "Write a short story about a librarian who discovers a magical book. The story should be exactly 100 words.",
        "Explain the concept of quantum entanglement in simple terms.",
        "Generate a Python function that takes a list of integers and returns a new list with only the even numbers."
    ]
    
    results = []
    print(f"\nRunning {len(sample_prompts)} sample prompts...")
    for i, prompt in enumerate(sample_prompts):
        print(f"  Querying prompt {i+1}/{len(sample_prompts)}...")
        result = query_token_optimiser(prompt)
        if result:
            results.append(result)
            print(f"    -> Response generated in {result['total_time_sec']:.2f}s ({result['tokens_per_second']:.2f} TPS)")
        else:
            print("    -> Failed to get response. Aborting evaluation for this model.")
            return None
    
    # --- Aggregate Performance Metrics ---
    total_prompt_tokens = sum(r['prompt_tokens'] for r in results)
    total_completion_tokens = sum(r['completion_tokens'] for r in results)
    avg_latency = sum(r['total_time_sec'] for r in results) / len(results)
    avg_tps = sum(r['tokens_per_second'] for r in results) / len(results)

    # --- Cost Calculation (C_m(p)) ---
    # This step calculates the total cost for all inference attempts.
    total_hypothetical_cost = calculate_inference_cost(
        total_prompt_tokens, model_config['hypothetical_cost_per_input_token'],
        total_completion_tokens, model_config['hypothetical_cost_per_output_token']
    )
    
    # --- Get Manual Success Rate (R_m(p)) ---
    # The paper uses pass@1 (solving the problem in one attempt) to measure
    # effectiveness. This is the success rate, R_m(p). Here, we
    # manually assess the quality of the generated responses to determine this rate.
    while True:
        try:
            success_rate_input = input("\nBased on the quality of the responses, what is the success rate (pass@1)? "
                                       "Enter a percentage (e.g., 75): ")
            success_rate = float(success_rate_input) / 100
            if 0.0 <= success_rate <= 1.0:
                break
            else:
                print("Please enter a number between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # --- Final Cost-of-Pass Calculation (v(m,p)) ---
    # This final step combines the total cost and success rate to get the
    # comprehensive cost-of-pass metric.
    cost_of_pass = calculate_cost_of_pass(total_hypothetical_cost, success_rate)
    
    # --- Memory Measurement (After) ---
    mem_after = get_gpu_memory_utilization()
    model_vram_footprint = (mem_after['used_mib'] - mem_before['used_mib']) if mem_before and mem_after else 0

    # --- Store all results ---
    final_report = {
        "Model Name": model_config['name'],
        "Avg Latency (s)": f"{avg_latency:.2f}",
        "Avg TPS": f"{avg_tps:.2f}",
        "VRAM Footprint (MiB)": f"{model_vram_footprint:.2f}",
        "Success Rate": f"{success_rate:.1%}",
        "Hypothetical Cost ($)": f"{total_hypothetical_cost:.6f}",
        "Cost-of-Pass ($)": f"{cost_of_pass:.6f}" if cost_of_pass != math.inf else "inf"
    }
    
    print("\n--- Evaluation Summary ---")
    for key, value in final_report.items():
        print(f"{key:<25}: {value}")
    
    return final_report

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --- Model Configurations ---
    # Define the two models you want to evaluate.
    # The cost-per-token values are hypothetical, used only for comparison.
    # You can find common pricing from major providers to use as a baseline.
    MODEL_1_CONFIG = {
        "name": "liquid/lfm2-1.2b",
        "hypothetical_cost_per_input_token": 0.20 / 1_000_000,
        "hypothetical_cost_per_output_token": 1.00 / 1_000_000
    }

    MODEL_2_CONFIG = {
        "name": "qwen.qwen3-4b-instruct-2507",
        "hypothetical_cost_per_input_token": 0.20 / 1_000_000,
        "hypothetical_cost_per_output_token": 1.00 / 1_000_000
    }

    models_to_evaluate = [MODEL_1_CONFIG, MODEL_2_CONFIG]
    all_results = []
    
    # Initialize LM Studio Client
    lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    
    # Initialize NVML for memory reading
    if pynvml:
        pynvml.nvmlInit()
    
    try:
        for config in models_to_evaluate:
            report = evaluate_model(config, lm_studio_client)
            if report:
                all_results.append(report)
    finally:
        # Ensure NVML is shut down properly
        if pynvml:
            pynvml.nvmlShutdown()

    # --- Final Comparison Table ---
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("Final Comparison Report")
        print("="*80)
        
        headers = list(all_results[0].keys())
        # Print header
        header_str = f"{headers[0]:<55}" + "".join([f"{h:<25}" for h in headers[1:]])
        print(header_str)
        print("-" * len(header_str))

        # Print rows
        for report in all_results:
            row_str = f"{report[headers[0]]:<55}" + "".join([f"{str(report[h]):<25}" for h in headers[1:]])
            print(row_str)