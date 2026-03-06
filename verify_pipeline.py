import os
import sys
import time

# Add root to path
sys.path.append(os.getcwd())

from doc_storage.document_store import ingest
from orchestrator.pipeline import run_pipeline

def run_test():
    print("🚀 Starting End-to-End Pipeline Verification\n")

    # 1. Create a dummy test file
    test_file = "test_doc.txt"
    with open(test_file, "w") as f:
        f.write("The capital of France is Paris. It is known for the Eiffel Tower.\n")
        f.write("The Token Gater project uses entropy to minimize context windows.\n")
        f.write("Gemini CLI is an interactive agent for software engineering.\n")

    print(f"📝 Created test file: {test_file}")

    # 2. Ingest document
    print("📥 Ingesting document into Redis...")
    try:
        stored = ingest(test_file)
        print(f"✅ Ingested: {stored}")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return

    # 3. Run Pipeline with Gating
    query = "What is the capital of France and what does the Token Gater project use?"
    print(f"\n🔍 Running pipeline with query: '{query}'")
    print("⚙️  Mode: entropy")
    
    try:
        result = run_pipeline(query, gating_mode="entropy")
        
        print("\n--- Pipeline Result ---")
        print(f"Response: {result['gated']['response_text']}")
        print(f"Prompt Tokens: {result['gated']['prompt_tokens']}")
        print(f"Time: {result['gated']['total_time_sec']:.2f}s")
        print(f"Gating Stats: {result['gating_stats']}")
        print("-----------------------\n")
        
        if result['gated']['response_text']:
            print("✅ Pipeline executed successfully.")
        else:
            print("⚠️ Pipeline executed but response text is empty.")

    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    run_test()
