import json
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from ...rag.document_ingestor import chunk_text # Assuming chunk_text is usable
from ...config.settings import EMBED_MODEL

class SessionDocumentStore:
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.model = SentenceTransformer(EMBED_MODEL)

    def add_document(self, session_id: str, document_content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # Chunk the text
        chunks = chunk_text(document_content)
        
        # Generate embeddings
        embeddings = [self.model.encode(c).tolist() for c in chunks]

        for chunk, embedding in zip(chunks, embeddings):
            self.sessions[session_id].append({
                "chunk": chunk,
                "embedding": embedding
            })
        return len(chunks)

    def get_documents(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

# Singleton instance
session_document_store = SessionDocumentStore()
