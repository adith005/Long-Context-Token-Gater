import streamlit as st
import asyncio
import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.evaluator import query_token_optimiser, calculate_inference_cost, calculate_cost_of_pass

# --- Page Setup ---
st.set_page_config(
    page_title="Token Optimizer", 
    layout="wide"
)

st.title("⚔️ Token Optimizer Playground")
st.markdown("Enter a prompt below to compare the performance of the gating and non-gating LLMs.")

# --- Main Execution Loop ---
async def run_comparison(prompt, placeholder_gated, placeholder_non_gated, results_gated, results_non_gated):
    """
    Creates two asynchronous tasks to run in parallel.
    """
    gated_task = asyncio.to_thread(query_token_optimiser, prompt, gating_mode="original_gating")
    non_gated_task = asyncio.to_thread(query_token_optimiser, prompt, gating_mode=None)

    gated_response, non_gated_response = await asyncio.gather(gated_task, non_gated_task)

    # Display gated results
    with placeholder_gated.container():
        st.subheader("Gating LLM")
        st.markdown(gated_response['response_text'])
        gated_df = pd.DataFrame([gated_response])
        gated_df['hypothetical_cost'] = calculate_inference_cost(gated_response['prompt_tokens'], 0.20 / 1_000_000, gated_response['completion_tokens'], 1.00 / 1_000_000)
        gated_df['cost_of_pass'] = calculate_cost_of_pass(gated_df['hypothetical_cost'], 1.0)
        st.dataframe(gated_df[['prompt_tokens', 'completion_tokens', 'total_time_sec', 'tokens_per_second', 'hypothetical_cost', 'cost_of_pass']])

    # Display non-gated results
    with placeholder_non_gated.container():
        st.subheader("Non-Gating LLM")
        st.markdown(non_gated_response['response_text'])
        non_gated_df = pd.DataFrame([non_gated_response])
        non_gated_df['hypothetical_cost'] = calculate_inference_cost(non_gated_response['prompt_tokens'], 0.20 / 1_000_000, non_gated_response['completion_tokens'], 1.00 / 1_000_000)
        non_gated_df['cost_of_pass'] = calculate_cost_of_pass(non_gated_df['hypothetical_cost'], 1.0)
        st.dataframe(non_gated_df[['prompt_tokens', 'completion_tokens', 'total_time_sec', 'tokens_per_second', 'hypothetical_cost', 'cost_of_pass']])


# --- UI Layout ---

# 1. Create the input field at the top
user_input = st.chat_input("Enter your prompt for both models...")

# 2. Create the layout columns
col1, col2 = st.columns(2)

# 3. Create empty placeholders inside the columns
with col1:
    gated_output = st.empty()

with col2:
    non_gated_output = st.empty()

# Create placeholders for results
results_gated = st.empty()
results_non_gated = st.empty()

# 4. Handle Input
if user_input:
    # Show the user's prompt first
    st.info(f"**User:** {user_input}")
    
    # Run the async loop
    try:
        asyncio.run(run_comparison(user_input, gated_output, non_gated_output, results_gated, results_non_gated))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_comparison(user_input, gated_output, non_gated_output, results_gated, results_non_gated))