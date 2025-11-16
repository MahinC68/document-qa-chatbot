from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from config import get_chat_model, get_embeddings_model

VECTORSTORE_FOLDER = "vectorstore"

app = Flask(__name__)

def load_vectorstore():
    """Load FAISS vector store from local folder."""
    embeddings = get_embeddings_model()
    return FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )


# Load vectorstore at startup
vectorstore = load_vectorstore()

# Build retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=get_chat_model(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)


@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle JSON-based question requests."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Request must contain 'question' field."}), 400

    question = data["question"]
    print(f"Received question: {question}")

    try:
        answer = qa_chain.run(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Document Q&A Chatbot API is running."})


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
