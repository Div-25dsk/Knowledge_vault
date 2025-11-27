from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import json

# -----------------------------
# 1. Extract Text
# -----------------------------
def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# -----------------------------
# 2. Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# 3. Embeddings (TF-IDF)
# -----------------------------
def embed_chunks(chunks: list):
    # Remove empty or whitespace-only chunks
    clean_chunks = [c.strip() for c in chunks if c.strip()]
    if not clean_chunks:
        raise ValueError("No valid text chunks found to embed.")

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(clean_chunks)
    embedded = []
    for i, chunk in enumerate(clean_chunks):
        embedded.append({"chunk": chunk, "embeds": vectors[i].toarray()[0].tolist()})
    return embedded, vectorizer

# -----------------------------
# 4. Similarity Search
# -----------------------------
def search_similar_chunks(query: str, embedded_chunks: list, vectorizer, top_k: int = 3):
    query_vec = vectorizer.transform([query]).toarray()
    scores = []
    for item in embedded_chunks:
        chunk_vec = np.array(item["embeds"]).reshape(1, -1)
        score = cosine_similarity(query_vec, chunk_vec)[0][0]
        scores.append((score, item["chunk"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

# -----------------------------
# 5. Prepare Context
# -----------------------------
def prepare_context(retrieved_chunks):
    return "\n\n".join(chunk for _, chunk in retrieved_chunks)

# -----------------------------
# 6. Answer Generator (lightweight)
# -----------------------------
def generate_answer(query: str, context: str) -> str:
    # Instead of calling an LLM, just return the top chunks as the "answer"
    return f"Here are the most relevant sections for your query:\n\n{context}"

# -----------------------------
# 7. RUN RAG
# -----------------------------
def run_rag(query: str, embedded_chunks: list, vectorizer):
    top_chunks = search_similar_chunks(query, embedded_chunks, vectorizer)
    context = prepare_context(top_chunks)
    answer = generate_answer(query, context)
    return answer

# -----------------------------
# 8. CLASS WRAPPER
# -----------------------------
class RAGEngine:

    def store_embeddings(self, chunks, embeddings, file_name: str):
        os.makedirs("vector_db", exist_ok=True)
        data = []
        for item in embeddings:
            data.append({
                "chunk": item["chunk"],
                "embeds": item["embeds"]
            })
        with open(f"vector_db/{file_name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def load_embeddings(self, file_name: str):
        path = f"vector_db/{file_name}.json"
        if not os.path.exists(path):
            raise ValueError("No stored vectors found for this file.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def ingest_doc(self, pdf_path: str, file_name: str):
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings, vectorizer = embed_chunks(chunks)
        self.store_embeddings(chunks, embeddings, file_name)
        return True

    def query_doc(self, query_text: str, file_name: str):
        embedded_chunks = self.load_embeddings(file_name)
        chunks = [item["chunk"] for item in embedded_chunks]
        _, vectorizer = embed_chunks(chunks)
        top_chunks = search_similar_chunks(query_text, embedded_chunks, vectorizer)
        context = prepare_context(top_chunks)
        answer = generate_answer(query_text, context)
        return answer
