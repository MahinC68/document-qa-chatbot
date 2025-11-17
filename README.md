📄 Document Q&A Chatbot

A lightweight Retrieval-Augmented Generation (RAG) system that allows users to upload a PDF and ask questions about its content through a Flask API.
The backend automatically ingests documents, generates embeddings, stores them in a FAISS vector database, and uses GPT-4o-mini to answer queries based on relevant retrieved chunks.

🚀 Features

PDF Ingestion Pipeline
Automatically loads PDFs, chunks text, generates embeddings, and stores them in FAISS.

Vector Database (FAISS)
Fast similarity search for retrieving the most relevant sections of the document.

OpenAI Chat Model (GPT-4o-mini)
Used for generating context-aware, high-quality answers.

REST API (Flask)
Query the document using a simple /ask endpoint with JSON input.

Modular Architecture
Clear separation of ingestion, configuration, and Q&A logic.

🛠️ Tech Stack

Python 3.12

Flask

LangChain

OpenAI API (GPT-4o-mini + Embeddings)

FAISS

dotenv

📂 Project Structure

document-qa-chatbot/
│── app.py               # Flask API
│── ingest.py            # Document ingestion + vectorstore builder
│── config.py            # API key + embeddings setup
│── requirements.txt     # Dependencies
│── .gitignore
│── .env                 # (not committed) API key
│── docs/                # PDFs go here
│── vectorstore/         # Generated FAISS index
│── venv/                # Virtual environment

⚙️ Setup Instructions

1. Clone the repo
git clone <your-repo-link>
cd document-qa-chatbot

2. Install dependencies
pip install -r requirements.txt

3. Add your OpenAI API key
Create a .env file: OPENAI_API_KEY=your_key_here

📥 Ingest a PDF

Place your PDF inside the docs/ folder
Run: py ingest.py

If successful, you will see:
Vector store saved to: vectorstore/
Ingestion complete!

🤖 Run the Chatbot API
py app.py

You should see:
Running on http://127.0.0.1:5000

💬 Ask a Question
Use a REST client (e.g., VS Code REST Client, Postman, or curl).

Example Request:

POST http://127.0.0.1:5000/ask
Content-Type: application/json

{
  "question": "Give me a short summary of the document."
}

Example Response:

{
  "answer": "This document discusses..."
}

🔄 Updating the Document

To use a new PDF:

Replace the old file in docs/

Delete the vectorstore/ folder (optional)

Re-run: py ingest.py

Restart: py app.py

📌 Notes

.env file is intentionally excluded from GitHub for security

FAISS vectorstore updates only when ingestion is re-run

⭐ Future Improvements

Add a web frontend (React/HTML)

Add multi-document support

Add file upload API endpoint

Deploy to cloud (Render/Railway/Azure)