import streamlit as st
import asyncio
import sys
import os
import json
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.pipeline import run_pipeline
from doc_storage.document_store import ingest, list_registry
from memory_storage.storage import dedup_memory, expire_stale_memory
from needlebench import run_benchmark, NEEDLES, HAYSTACK_SIZES
from tracer import PipelineTracer


# ─────────────────────────────────────────────────────────────
# Cost & Metric Helpers
# ─────────────────────────────────────────────────────────────

INPUT_COST_PER_TOKEN  = 0.20 / 1_000_000   # $0.20 per 1M input tokens
OUTPUT_COST_PER_TOKEN = 1.00 / 1_000_000   # $1.00 per 1M output tokens

def calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens     * INPUT_COST_PER_TOKEN +
        completion_tokens * OUTPUT_COST_PER_TOKEN,
        8,
    )

def calc_tps(completion_tokens: int, total_time_sec: float) -> float:
    if total_time_sec and total_time_sec > 0:
        return round(completion_tokens / total_time_sec, 2)
    return 0.0

def build_metrics_df(result_dict: dict) -> pd.DataFrame:
    """Build full metrics dataframe from a pipeline result."""
    g  = result_dict.get("gated", {})
    gs = result_dict.get("gating_stats", {})

    pt  = g.get("prompt_tokens",     0)
    ct  = g.get("completion_tokens", 0)
    ts  = g.get("total_time_sec",    0) or 0

    row = {
        "prompt_tokens":      pt,
        "completion_tokens":  ct,
        "total_time_sec":     round(ts, 3),
        "tokens_per_second":  calc_tps(ct, ts),
        "hypothetical_cost":  f"${calc_cost(pt, ct):.8f}",
        "cost_of_pass":       f"${calc_cost(pt, ct):.8f}",
    }

    if gs:
        row["candidates_in"]  = gs.get("candidates_in", "—")
        row["window_size"]    = gs.get("window_size",   "—")
        row["pruned"]         = gs.get("pruned",        "—")
        row["plateau_at"]     = gs.get("plateau_at",    "—")
        row["window_entropy"] = result_dict.get("window_entropy", "—")
        row["entropy_stable"] = result_dict.get("is_stable",      "—")

    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────
# Page Setup
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Token Optimizer", layout="wide")
st.title("⚔️ Token Optimizer Playground")

tab_playground, tab_trace, tab_needlebench = st.tabs([
    "💬 Playground",
    "🔍 Pipeline Trace",
    "🔬 NeedleBench Eval",
])


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF to add to the knowledge base.", type="pdf"
)

if uploaded_file is not None:
    temp_dir  = "tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        stored_docs = ingest(temp_path)
        st.sidebar.success(f"Ingested: {', '.join(stored_docs)}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ── Memory Deduplication ──────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🧹 Memory Dedup")
dedup_threshold = st.sidebar.slider(
    "Merge threshold", min_value=0.70, max_value=0.99,
    value=0.85, step=0.01,
    help="Entries with cosine similarity above this are merged. "
         "0.97+ = exact duplicates only. 0.85 = same topic, different wording."
)
if st.sidebar.button("Run Dedup Sweep"):
    with st.sidebar:
        with st.spinner("Scanning memory…"):
            stats = dedup_memory(threshold=dedup_threshold)
        st.success(
            f"Done — scanned {stats['scanned']}, "
            f"merged {stats['merged']}, "
            f"deleted {stats['deleted']}, "
            f"{stats['remaining']} remaining"
        )

st.sidebar.divider()
st.sidebar.header("⏳ Expire Stale Memories")
ttl_days = st.sidebar.number_input(
    "Max age (days)", min_value=1, max_value=365, value=30,
    help="Entries older than this AND below the usefulness floor will be deleted."
)
ttl_usefulness = st.sidebar.slider(
    "Usefulness floor", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Entries with usefulness below this are candidates for expiry."
)
if st.sidebar.button("Run Expiry"):
    with st.sidebar:
        with st.spinner("Expiring stale memories…"):
            estats = expire_stale_memory(ttl_days=ttl_days, usefulness_floor=ttl_usefulness)
        st.success(f"Done — scanned {estats['scanned']}, expired {estats['expired']}")
st.sidebar.divider()
st.sidebar.header("📚 Ingested Documents")

registry = list_registry()
if not registry:
    st.sidebar.caption("No documents ingested yet.")
else:
    for entry in registry:
        with st.sidebar.expander(f"📄 {entry['original_filename']}"):
            st.caption(f"Sentences: {entry['sentence_count']}  |  {entry['ingested_at']}")
            st.markdown(f"**Summary**")
            st.write(entry["summary"])
            st.caption(f"`{entry['json_path']}`")


# ─────────────────────────────────────────────────────────────
# Shared: render a trace event as an expander
# ─────────────────────────────────────────────────────────────

STEP_ICONS = {
    1: "📥",   # Input
    2: "🔢",   # Embed
    3: "🔍",   # Retrieve
    4: "🚪",   # Gate
    5: "📝",   # Prompt
    6: "🤖",   # LLM
    7: "💾",   # Memory
    8: "📤",   # Output
}

def render_event(event: dict, container=None):
    target = container or st
    icon   = STEP_ICONS.get(event["step"], "•")
    status = "✅" if event["status"] == "ok" else "❌"
    label  = f"{status} Step {event['step']} — {icon} {event['name']}  (+{event['elapsed']}s)"

    with target.expander(label, expanded=False):
        st.json(event["data"])


def render_trace(events: list, container=None):
    target = container or st
    if not events:
        target.info("No trace events yet.")
        return
    for event in events:
        render_event(event, target)


# ─────────────────────────────────────────────────────────────
# TAB 1 — Playground
# ─────────────────────────────────────────────────────────────

with tab_playground:
    st.markdown("Compare gated vs non-gated responses side by side.")

    async def run_comparison(prompt, ph_gated, ph_non_gated):
        gated_task     = asyncio.to_thread(run_pipeline, prompt, gating_mode="entropy")
        non_gated_task = asyncio.to_thread(run_pipeline, prompt, gating_mode="none")
        gated_result, non_gated_result = await asyncio.gather(gated_task, non_gated_task)

        # ── Gated column ──────────────────────────────────────
        with ph_gated.container():
            st.subheader("🔒 Gating LLM (entropy)")
            st.markdown(gated_result["gated"]["response_text"])

            with st.expander("📊 Pipeline Trace — Gated", expanded=True):
                st.dataframe(build_metrics_df(gated_result), use_container_width=True)

                g  = gated_result.get("gated", {})
                gs = gated_result.get("gating_stats", {})
                pt = g.get("prompt_tokens", 0)
                ct = g.get("completion_tokens", 0)
                ts = g.get("total_time_sec", 0) or 0

                m1, m2, m3 = st.columns(3)
                m1.metric("Prompt Tokens",     pt)
                m2.metric("Completion Tokens", ct)
                m3.metric("Time (s)",          round(ts, 3))

                m4, m5, m6 = st.columns(3)
                m4.metric("Tokens / sec",      calc_tps(ct, ts))
                m5.metric("Hypothetical Cost", f"${calc_cost(pt, ct):.8f}")
                m6.metric("Window Size",       gs.get("window_size", "—"))

                if gs:
                    st.caption(
                        f"Candidates in: {gs.get('candidates_in','—')}  |  "
                        f"Pruned: {gs.get('pruned','—')}  |  "
                        f"Plateau at: {gs.get('plateau_at','—')}  |  "
                        f"Entropy stable: {gated_result.get('is_stable','—')}"
                    )

        # ── Non-gated column ──────────────────────────────────
        with ph_non_gated.container():
            st.subheader("🔓 Non-Gating LLM (none)")
            st.markdown(non_gated_result["gated"]["response_text"])

            with st.expander("📊 Pipeline Trace — Non-Gated", expanded=True):
                st.dataframe(build_metrics_df(non_gated_result), use_container_width=True)

                g  = non_gated_result.get("gated", {})
                pt = g.get("prompt_tokens", 0)
                ct = g.get("completion_tokens", 0)
                ts = g.get("total_time_sec", 0) or 0

                m1, m2, m3 = st.columns(3)
                m1.metric("Prompt Tokens",     pt)
                m2.metric("Completion Tokens", ct)
                m3.metric("Time (s)",          round(ts, 3))

                m4, m5, m6 = st.columns(3)
                m4.metric("Tokens / sec",      calc_tps(ct, ts))
                m5.metric("Hypothetical Cost", f"${calc_cost(pt, ct):.8f}")
                m6.metric("Window Size",       "all candidates")

        # ── Cross-comparison summary ───────────────────────────
        g_pt  = gated_result.get("gated", {}).get("prompt_tokens", 0)
        ng_pt = non_gated_result.get("gated", {}).get("prompt_tokens", 0)
        g_ct  = gated_result.get("gated", {}).get("completion_tokens", 0)
        ng_ct = non_gated_result.get("gated", {}).get("completion_tokens", 0)

        g_cost  = calc_cost(g_pt,  g_ct)
        ng_cost = calc_cost(ng_pt, ng_ct)
        tcr     = round(ng_pt / g_pt, 4) if g_pt > 0 else 1.0
        saved   = round(ng_cost - g_cost, 8)

        st.divider()
        st.subheader("⚖️ Comparison Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Token Compression Ratio", tcr,
            delta=f"{round((tcr-1)*100, 1)}% fewer tokens" if tcr > 1 else None,
        )
        c2.metric(
            "Cost saved per query", f"${saved}",
            delta=f"{round((1 - g_cost/ng_cost)*100, 2)}% cheaper" if ng_cost > 0 else None,
        )
        c3.metric(
            "Gated window size",
            gated_result.get("gating_stats", {}).get("window_size", "—"),
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
# TAB 2 — Pipeline Trace
# ─────────────────────────────────────────────────────────────

with tab_trace:
    st.markdown(
        "Run a query and watch every pipeline step in real time — "
        "inputs, outputs, timing, and what the gater selected."
    )

    tc1, tc2 = st.columns([3, 1])
    with tc1:
        trace_query = st.text_input(
            "Query", placeholder="e.g. What is the operating frequency of the SX1278?"
        )
    with tc2:
        trace_mode = st.selectbox("Gating mode", ["entropy", "simple", "none", "bm25", "joint", "quantum", "hybrid"], key="trace_mode")

    run_trace_btn = st.button("▶ Run & Trace", type="primary")

    if run_trace_btn and trace_query:

        tracer   = PipelineTracer()
        trace_ph = st.empty()         # live update target
        done_ph  = st.empty()

        # We run the pipeline in a thread so Streamlit doesn't block,
        # but since PipelineTracer is synchronous we poll after completion.
        with st.spinner("Running pipeline…"):
            result = run_pipeline(trace_query, gating_mode=trace_mode, tracer=tracer)

        # ── Render all events ─────────────────────────────────
        st.success(f"Pipeline complete in {tracer.events[-1]['elapsed']}s")

        for event in tracer.events:
            render_event(event)

        # ── Final response ────────────────────────────────────
        st.divider()
        st.subheader("🤖 Final Response")
        st.markdown(result["gated"]["response_text"] or "*No response (LLM not connected?)*")

        g  = result.get("gated", {})
        gs = result.get("gating_stats", {})
        pt = g.get("prompt_tokens",     0)
        ct = g.get("completion_tokens", 0)
        ts = g.get("total_time_sec",    0) or 0

        st.subheader("📊 Output Metrics")
        st.dataframe(build_metrics_df(result), use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prompt Tokens",     pt)
        col_b.metric("Completion Tokens", ct)
        col_c.metric("Latency (s)",       round(ts, 3))

        col_d, col_e, col_f = st.columns(3)
        col_d.metric("Tokens / sec",      calc_tps(ct, ts))
        col_e.metric("Hypothetical Cost", f"${calc_cost(pt, ct):.8f}")
        col_f.metric("Cost of Pass",      f"${calc_cost(pt, ct):.8f}")

        if gs:
            col_g, col_h, col_i = st.columns(3)
            col_g.metric("Window Size",    gs.get("window_size",   "—"))
            col_h.metric("Candidates In",  gs.get("candidates_in", "—"))
            col_i.metric("Pruned",         gs.get("pruned",        "—"))
            st.caption(
                f"Plateau at: {gs.get('plateau_at','—')}  |  "
                f"Window entropy: {result.get('window_entropy','—')}  |  "
                f"Entropy stable: {result.get('is_stable','—')}"
            )

        # ── Raw trace download ────────────────────────────────
        st.download_button(
            label     = "⬇ Download trace JSON",
            data      = json.dumps(tracer.events, indent=2),
            file_name = "pipeline_trace.json",
            mime      = "application/json",
        )

    elif run_trace_btn and not trace_query:
        st.warning("Enter a query first.")


# ─────────────────────────────────────────────────────────────
# TAB 3 — NeedleBench Eval
# ─────────────────────────────────────────────────────────────

with tab_needlebench:
    st.markdown(
        "**NeedleBench** measures how well each gating strategy retrieves a specific "
        "fact buried inside a large set of distractors."
    )

    cfg1, cfg2, cfg3 = st.columns(3)

    with cfg1:
        selected_modes = st.multiselect(
            "Gating modes",
            options=["entropy", "simple", "none", "bm25", "joint", "quantum", "hybrid"],
            default=["entropy", "simple", "none", "bm25", "joint", "quantum", "hybrid"],
        )
    with cfg2:
        selected_haystacks = st.multiselect(
            "Haystack sizes",
            options=list(HAYSTACK_SIZES.keys()),
            default=list(HAYSTACK_SIZES.keys()),
            help="short=15, medium=40, long=80 fillers",
        )
    with cfg3:
        use_llm = st.toggle(
            "Include LLM calls", value=False,
            help="Requires LM Studio on localhost:1234. Off = retrieval-only (fast).",
        )

    n_tests = len(NEEDLES) * len(selected_modes) * len(selected_haystacks)
    st.caption(f"Tests to run: **{n_tests}**")

    run_nb_btn = st.button(
        "▶ Run NeedleBench", type="primary",
        disabled=not selected_modes or not selected_haystacks,
    )

    if run_nb_btn:
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

        st.subheader("Recall Rate by Haystack Size")
        recall_h = {
            mode: {h: summary[mode]["by_haystack"].get(h, {}).get("recall_rate", 0)
                   for h in selected_haystacks}
            for mode in selected_modes
        }
        st.dataframe(
            pd.DataFrame(recall_h).T.style.format("{:.1%}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
        )

        st.subheader("Recall Rate by Needle Depth")
        depths = ["shallow", "middle", "deep"]
        recall_d = {
            mode: {d: summary[mode]["by_depth"].get(d, {}).get("recall_rate", 0)
                   for d in depths}
            for mode in selected_modes
        }
        st.dataframe(
            pd.DataFrame(recall_d).T.style.format("{:.1%}").background_gradient(
                cmap="RdYlGn", vmin=0, vmax=1),
            use_container_width=True,
        )

        st.subheader("Avg Prompt Tokens by Haystack Size")
        token_data = {
            mode: {h: int(summary[mode]["by_haystack"].get(h, {}).get("avg_prompt_tokens", 0))
                   for h in selected_haystacks}
            for mode in selected_modes
        }
        st.dataframe(pd.DataFrame(token_data).T, use_container_width=True)

        with st.expander("Full result log"):
            st.dataframe(pd.DataFrame(bench_output["all_results"]), use_container_width=True)

        st.download_button(
            label     = "⬇ Download results JSON",
            data      = json.dumps(bench_output, indent=2),
            file_name = "needlebench_results.json",
            mime      = "application/json",
        )