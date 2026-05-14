# run with: uvicorn main:app --reload --port 8000

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import DOCS_FOLDER
from ingest import ingest_pdf

app = FastAPI(title="DocuChat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DOCS_FOLDER, exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(DOCS_FOLDER, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = ingest_pdf(filepath=save_path, filename=file.filename)

    return {"message": "success", "filename": file.filename, "chunks": chunks}


@app.post("/ask")
async def ask_question(body: dict):
    # placeholder — RAG logic goes here once the frontend is wired up
    return {"answer": "coming soon", "sources": []}
