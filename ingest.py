import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import get_embeddings_model


DOCS_FOLDER = "docs"
VECTORSTORE_FOLDER = "vectorstore"

def load_documents():
    """Load all PDF files from the docs/ folder."""
    documents = []

    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_FOLDER, filename)
            print(f"Loading: {filepath}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

    if not documents:
        raise RuntimeError("No PDF documents found in docs/ folder.")

    return documents


def split_documents(documents):
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Embed chunks using Azure OpenAI and save to FAISS."""
    embeddings = get_embeddings_model()

    print("Generating embeddings and building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_FOLDER)
    print(f"Vector store saved to: {VECTORSTORE_FOLDER}/")


def main():
    print("Step 1: Loading documents")
    docs = load_documents()

    print("Step 2: Splitting into chunks")
    chunks = split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    print("Step 3: Building FAISS vector store")
    build_vectorstore(chunks)

    print("Ingestion complete! You can now run the chatbot.")


if __name__ == "__main__":
    main()
