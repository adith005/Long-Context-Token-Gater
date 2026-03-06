import streamlit as st
import asyncio
import sys
import os
import json
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.pipeline import run_pipeline
from doc_storage.document_store import ingest
from needlebench import run_benchmark, NEEDLES, HAYSTACK_SIZES


# ─────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Token Optimizer", layout="wide")
st.title("⚔️ Token Optimizer Playground")

tab_playground, tab_needlebench = st.tabs(["💬 Playground", "🔬 NeedleBench Eval"])


# ─────────────────────────────────────────────────────────────
# Sidebar — PDF upload (shared across tabs)
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF document to add it to the knowledge base.",
    type="pdf"
)

if uploaded_file is not None:
    temp_dir = "tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        stored_docs = ingest(temp_path)
        st.sidebar.success(f"Ingested: {', '.join(stored_docs)}")
    except Exception as e:
        st.sidebar.error(f"Error ingesting PDF: {e}")


# ─────────────────────────────────────────────────────────────
# TAB 1 — Playground (original, unchanged)
# ─────────────────────────────────────────────────────────────

with tab_playground:
    st.markdown("Enter a prompt below to compare gated vs non-gated LLM responses.")

    async def run_comparison(prompt, placeholder_gated, placeholder_non_gated):
        gated_task     = asyncio.to_thread(run_pipeline, prompt, gating_mode="entropy")
        non_gated_task = asyncio.to_thread(run_pipeline, prompt, gating_mode="none")
        gated_result, non_gated_result = await asyncio.gather(gated_task, non_gated_task)

        with placeholder_gated.container():
            st.subheader("Gating LLM (entropy)")
            st.markdown(gated_result["gated"]["response_text"])
            st.dataframe(
                pd.DataFrame([gated_result["gated"]])[
                    ["prompt_tokens", "completion_tokens", "total_time_sec"]
                ]
            )

        with placeholder_non_gated.container():
            st.subheader("Non-Gating LLM (none)")
            st.markdown(non_gated_result["gated"]["response_text"])
            st.dataframe(
                pd.DataFrame([non_gated_result["gated"]])[
                    ["prompt_tokens", "completion_tokens", "total_time_sec"]
                ]
            )

    user_input = st.chat_input("Enter your prompt for both models...")
    col1, col2 = st.columns(2)
    with col1:
        gated_output = st.empty()
    with col2:
        non_gated_output = st.empty()

    if user_input:
        st.info(f"**User:** {user_input}")
        try:
            asyncio.run(run_comparison(user_input, gated_output, non_gated_output))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_comparison(user_input, gated_output, non_gated_output))


# ─────────────────────────────────────────────────────────────
# TAB 2 — NeedleBench Evaluation
# ─────────────────────────────────────────────────────────────

with tab_needlebench:
    st.markdown(
        "**NeedleBench** measures how well each gating strategy retrieves a specific "
        "fact buried inside a large set of distractors, and how accurately the LLM "
        "answers using only the selected context window."
    )

    # ── Config ────────────────────────────────────────────────
    cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

    with cfg_col1:
        selected_modes = st.multiselect(
            "Gating modes to compare",
            options=["entropy", "simple", "none"],
            default=["entropy", "simple", "none"],
        )

    with cfg_col2:
        selected_haystacks = st.multiselect(
            "Haystack sizes",
            options=list(HAYSTACK_SIZES.keys()),
            default=list(HAYSTACK_SIZES.keys()),
            help="short=15 fillers, medium=40, long=80",
        )

    with cfg_col3:
        use_llm = st.toggle(
            "Include LLM calls",
            value=False,
            help="Requires LM Studio running on localhost:1234. "
                 "Off = retrieval-only scoring (much faster).",
        )

    n_tests = len(NEEDLES) * len(selected_modes) * len(selected_haystacks)
    st.caption(
        f"Tests to run: **{n_tests}** "
        f"({len(NEEDLES)} needles × {len(selected_modes)} modes × {len(selected_haystacks)} sizes)"
    )

    run_btn = st.button(
        "▶ Run NeedleBench", type="primary",
        disabled=not selected_modes or not selected_haystacks
    )

    # ── Run ───────────────────────────────────────────────────
    if run_btn:
        progress_bar = st.progress(0, text="Starting…")
        live_log     = st.empty()
        live_rows    = []

        def on_progress(current, total, result):
            progress_bar.progress(
                current / total,
                text=(
                    f"[{current}/{total}]  {result.gating_mode} | "
                    f"{result.haystack_size} | {result.needle_id} — "
                    f"recall={'✅' if result.needle_recalled else '❌'}  "
                    f"score={result.answer_score:.2f}  tokens≈{result.prompt_tokens}"
                ),
            )
            live_rows.append({
                "mode":     result.gating_mode,
                "haystack": result.haystack_size,
                "needle":   result.needle_id,
                "depth":    result.depth,
                "recalled": "✅" if result.needle_recalled else "❌",
                "score":    result.answer_score,
                "tokens":   result.prompt_tokens,
                "win_sz":   result.window_size,
            })
            live_log.dataframe(pd.DataFrame(live_rows), use_container_width=True)

        with st.spinner("Running benchmark…"):
            bench_output = run_benchmark(
                modes          = selected_modes,
                haystack_sizes = selected_haystacks,
                call_llm_flag  = use_llm,
                verbose        = False,
                progress_cb    = on_progress,
            )

        progress_bar.progress(1.0, text="Complete ✅")
        summary = bench_output["summary"]

        # ── Overall comparison ────────────────────────────────
        st.subheader("Overall Results")
        overall_rows = []
        for mode in selected_modes:
            s = summary[mode]["overall"]
            overall_rows.append({
                "Gating Mode":    mode,
                "Recall Rate":    f"{s['recall_rate']:.1%}",
                "Avg Score":      f"{s['avg_answer_score']:.3f}",
                "Avg Tokens":     int(s["avg_prompt_tokens"]),
                "Avg Window Sz":  s["avg_window_size"],
                "Avg Latency(s)": s["avg_latency_sec"],
            })
        st.dataframe(pd.DataFrame(overall_rows), use_container_width=True)

        # ── Recall heatmap: haystack size ─────────────────────
        st.subheader("Recall Rate by Haystack Size")
        recall_h = {
            mode: {
                h: summary[mode]["by_haystack"].get(h, {}).get("recall_rate", 0)
                for h in selected_haystacks
            }
            for mode in selected_modes
        }
        st.dataframe(
            pd.DataFrame(recall_h).T.style.format("{:.1%}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
        )

        # ── Recall heatmap: depth ─────────────────────────────
        st.subheader("Recall Rate by Needle Depth")
        depths = ["shallow", "middle", "deep"]
        recall_d = {
            mode: {
                d: summary[mode]["by_depth"].get(d, {}).get("recall_rate", 0)
                for d in depths
            }
            for mode in selected_modes
        }
        st.dataframe(
            pd.DataFrame(recall_d).T.style.format("{:.1%}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
        )

        # ── Token cost ────────────────────────────────────────
        st.subheader("Avg Prompt Tokens by Haystack Size")
        token_data = {
            mode: {
                h: int(summary[mode]["by_haystack"].get(h, {}).get("avg_prompt_tokens", 0))
                for h in selected_haystacks
            }
            for mode in selected_modes
        }
        st.dataframe(pd.DataFrame(token_data).T, use_container_width=True)

        # ── Full log + download ───────────────────────────────
        with st.expander("Full result log"):
            st.dataframe(
                pd.DataFrame(bench_output["all_results"]),
                use_container_width=True,
            )

        st.download_button(
            label     = "⬇ Download results JSON",
            data      = json.dumps(bench_output, indent=2),
            file_name = "needlebench_results.json",
            mime      = "application/json",
        )
