# Token Optimiser v1.5
Multi-source context engine integrating Redis memory (MCP) + RAG + Token Gating.

## Setup
1. Run Redis locally
   docker run -d --name redis -p 6379:6379 redis:latest
2. Install requirements
   pip install -r requirements.txt
3. Add your OpenAI key:
   export OPENAI_API_KEY="sk-..."
4. Run document ingestion
   python rag/document_ingestor.py path/to/your.pdf
5. Start main:
   python main.py
