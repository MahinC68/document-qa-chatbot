import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_FOLDER, get_embeddings_model


def ingest_pdf(filepath: str, filename: str) -> int:
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        # needed so the /ask route can cite which file an answer came from
        chunk.metadata["filename"] = filename

    embeddings = get_embeddings_model()

    if os.path.exists(VECTORSTORE_FOLDER):
        # add to existing index so previously uploaded docs stay searchable
        vectorstore = FAISS.load_local(
            VECTORSTORE_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_FOLDER)
    return len(chunks)
