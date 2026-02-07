<<<<<<< HEAD
import sys
from gating.token_gater import gate
from llm.llm_interface import build_prompt, query_llm
from evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        query = ' '.join(sys.argv[2:])
        if not query:
            print("Please provide a query for evaluation.")
            print("Usage: python main.py --evaluate <your query>")
            sys.exit(1)
        run_evaluation(query)
    else:
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
=======
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
>>>>>>> 07dbd231591e9d8949116c306e72c9c24b85e74d
