import os
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_FOLDER, get_embeddings_model


def ingest_pdf(filepath: str, filename: str) -> int:
    reader = PdfReader(filepath)
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"filename": filename, "page": i},
        )
        for i, page in enumerate(reader.pages)
    ]

    # drop image-only pages that yielded no text — they'd produce useless embeddings
    pages = [p for p in pages if p.page_content.strip()]

    print(f"[ingest] {filename}: {len(pages)} page(s) with extractable text")
    if not pages:
        raise ValueError(f"No text could be extracted from {filename} — may be a scanned/image PDF")
    print(f"[ingest] first page preview: {pages[0].page_content[:200]!r}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    print(f"[ingest] {filename}: {len(chunks)} chunks created")

    embeddings = get_embeddings_model()

    # Always build a fresh index — replaces any previous document so only
    # the current upload is ever searched.
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_FOLDER)
    return len(chunks)
