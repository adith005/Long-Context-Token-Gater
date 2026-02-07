from mcp import Provider, register_function, mcp_serve
from . import history_db

provider = Provider("memory")

@register_function(provider)
def list_memories(limit: int = 20):
    return history_db.retrieve_similar("", k=limit)

@register_function(provider)
def add_memory(user_query, assistant_response, tags=None):
    return history_db.create_entry(user_query, assistant_response, tags)

if __name__ == "__main__":
    mcp_serve(provider)
