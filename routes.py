from fastapi import APIRouter, UploadFile, File, Form
from app.utils import save_temp_file, upload_file_to_s3
from app.rag_engine import RAGEngine
import os
rag = RAGEngine()


router = APIRouter()

@router.post("/upload")
async def upload(file:UploadFile = File(...)):

    
    temp_path = save_temp_file(file)

    s3_url = upload_file_to_s3(temp_path)
    clean_name = os.path.splitext(file.filename.replace(" ", "_").lower())[0]

    
    rag.ingest_doc(temp_path, clean_name)

    return {"message": "File uploaded and ingested successfully", "s3_url": s3_url,"db_key": clean_name}

@router.post("/query")
async def query(q: str = Form(...),file_name: str = Form(...)):
    file_name = file_name.replace(" ", "_").lower()
    result = rag.query_doc(q,file_name)
    return {"query": q , "answer":result}