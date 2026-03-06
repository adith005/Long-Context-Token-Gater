from gating.token_gater import gate
from llm.llm_interface import build_prompt, query_llm

if __name__ == "__main__":
    while True:
        print("=== TOKEN OPTIMISER v1.5 ===")
        user_query = input("Enter your query: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        # 1️⃣ Gate and fetch context
        context = gate(user_query)

        # 2️⃣ Build optimized prompt
        prompt = build_prompt(context, user_query)

        # 3️⃣ Send to LLM
        print("\n🧠 Generating response...")
        answer = query_llm(prompt)
        print("\nAssistant:", answer)
