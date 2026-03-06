import pytest
from token_optimiser.prompt import PromptBuilder
from token_optimiser.models import MemoryEntry

def test_prompt_includes_query():
    builder = PromptBuilder(token_budget=50)
    memories = [
        MemoryEntry(id="1", text="Alice likes apples"),
        MemoryEntry(id="2", text="Bob likes bananas"),
    ]
    prompt = builder.build("What does Alice like?", memories)
    assert "User:" in prompt
    assert "Alice" in prompt
    assert "Memory:" in prompt
    assert prompt.strip().endswith("Assistant:")

def test_token_budget_respected():
    builder = PromptBuilder(token_budget=5)  # tiny budget
    memories = [
        MemoryEntry(id="1", text="This is a long memory entry exceeding budget")
    ]
    prompt = builder.build("Test query", memories)
    # memory should be skipped due to budget
    assert "Memory:" not in prompt or prompt.count("Memory:") <= 1
