import requests
import json
from config.settings import LLM_MODEL

# LM Studio default endpoint
LMSTUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

def build_prompt(contexts, query):
    prompt = "### Context Memory ###\n"
    for i, c in enumerate(contexts):
        prompt += f"- Context {i+1}: {c}\n"
    prompt += f"\n### User Query ###\n{query}\n\n### Assistant ###\n"
    return prompt


def query_llm(prompt):
    """
    Send the prompt to LM Studio's REST API using a local model.
    """

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": LLM_MODEL,   # e.g. "liquid/lfm2-1.2b"
        "messages": [
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(LMSTUDIO_API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LM Studio error {response.status_code}: {response.text}")
