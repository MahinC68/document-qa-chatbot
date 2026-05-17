# run with: uvicorn main:app --reload --port 8000

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from config import DOCS_FOLDER
from ingest import ingest_pdf
from rag import session_memory
from agent import run_agent

app = FastAPI(title="DocuChat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://docuchat-frontend-bgnc.onrender.com",
    ],
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


class AskRequest(BaseModel):
    question: str
    session_id: str


@app.post("/ask")
async def ask_question(body: AskRequest):
    try:
        # Route through the LangGraph agent — it decides whether to use RAG or a direct LLM call
        return run_agent(question=body.question, session_id=body.session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Remove a session's conversation history so the next question starts fresh."""
    session_memory.pop(session_id, None)  # no-op if session never existed
    return {"message": "session cleared"}
