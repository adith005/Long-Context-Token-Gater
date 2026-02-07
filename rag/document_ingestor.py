import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from config.settings import EMBED_MODEL, DOC_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

model = SentenceTransformer(EMBED_MODEL)
os.makedirs(DOC_STORE_PATH, exist_ok=True)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
    chunks = chunk_text(text)
    embeddings = [model.encode(c).tolist() for c in chunks]

    doc_name = os.path.basename(pdf_path).split(".")[0]
    with open(f"{DOC_STORE_PATH}/{doc_name}.json", "w") as f:
        json.dump({"chunks": chunks, "embeddings": embeddings}, f)

    return len(chunks)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m rag.document_ingestor <path_to_pdf>")
    else:
        pdf_path = sys.argv[1]
        print(f"📄 Ingesting {pdf_path} ...")
        n_chunks = ingest_pdf(pdf_path)
        print(f"✅ Processed {n_chunks} text chunks and saved to rag_storage/")
