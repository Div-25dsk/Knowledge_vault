from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
import json
client = Groq(api_key=os.getenv("GROQ_API_KEY"))



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
# 3. Embeddings
# -----------------------------
def embed_chunks(chunks: list) -> list:
    embedded = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector = response.data[0].embedding
        embedded.append({"chunk": chunk, "embeds": vector})
    return embedded


# -----------------------------
# 4. Similarity Search
# -----------------------------
def search_similar_chunks(query: str, embedded_chunks: list, top_k: int = 3):
    
    query_vec = client.embeddings.create(model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores = []
    for item in embedded_chunks:
        chunk_vec = [item["embeds"]]
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
# 6. LLM Answer Generator
# -----------------------------
def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are an assistant who answers ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# -----------------------------
# 7. RUN RAG
# -----------------------------
def run_rag(query: str, embedded_chunks: list):
    top_chunks = search_similar_chunks(query, embedded_chunks)
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

        print("Embeddings stored successfully!")
        return True

    
    def load_embeddings(self, file_name: str):
        path = f"vector_db/{file_name}.json"
        if not os.path.exists(path):
            raise ValueError("No stored vectors found for this file.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def ingest_doc(self, pdf_path: str, file_name: str):
        print("Extracting text...")
        text = extract_text_from_pdf(pdf_path)

        print("Chunking...")
        chunks = chunk_text(text)

        print("Embedding...")
        embeddings = embed_chunks(chunks)

        print("Storing embeddings...")
        self.store_embeddings(chunks, embeddings, file_name)

        print("Ingestion complete.")
        return True

    
    def query_doc(self, query_text: str, file_name: str):
        print("Loading stored vectors...")
        embedded_chunks = self.load_embeddings(file_name)

        print("Searching similar chunks...")
        top_chunks = search_similar_chunks(query_text, embedded_chunks)

        print("Preparing context...")
        context = prepare_context(top_chunks)

        print("Generating answer...")
        answer = generate_answer(query_text, context)

        return answer
