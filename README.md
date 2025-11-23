# Knowledge Vault – RAG Backend (FastAPI + Groq)

A lightweight Retrieval-Augmented Generation (RAG) backend that:
- Extracts text from PDF files  
- Splits text into chunks  
- Creates embeddings using **Groq Embeddings API**  
- Performs similarity search using **cosine similarity**  
- Generates answers using **LLaMA-3 via Groq**  
- Exposes everything through **FastAPI API routes**

---

##  Tech Stack
- FastAPI  
- Groq API (Embeddings + LLaMA-3)  
- pypdf  
- scikit-learn  
- Python 3.10+  
- AWS EC2 (for deployment)  

---

## 📂 Project Structure
knowledge_vault/
│
├── app/
│ ├── main.py
│ ├── routes.py
│ ├── rag_engine.py
│ └── utils.py
│
├── vector_db/ # Stores JSON embedding files
├── temp/ # Temporary file uploads
├── requirements.txt
└── README.md


---

##  How to Run Locally

### 1. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

2. Install Dependencies
pip install -r requirements.txt

3. Start FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Visit:

http://localhost:8000/docs


Using the API
1. Ingest a PDF

Uploads the PDF → extracts text → chunks → creates embeddings → stores in vector_db.

2. Ask a Question

Sends your query → finds similar chunks → LLaMA-3 answers based on context.

👩‍💻 Author

Divya Bharathi D
Backend • Cloud • Data Enthusiast 


