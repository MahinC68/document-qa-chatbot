from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from config import get_embeddings_model

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

# Load vectorstore
vectorstore = load_vectorstore()

# Chat model (LLM)
llm = ChatOpenAI(model="gpt-4o-mini")

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LCEL pipeline for retrieval + answer generation
qa_chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | (lambda x: f"Use the context below to answer the question.\n\nContext:\n{x['context']}\n\nQuestion:\n{x['question']}")
    | llm
)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    
    if not data or "question" not in data:
        return jsonify({"error": "Request must contain 'question' field."}), 400
    
    question = data["question"]
    print(f"Received question: {question}")

    try:
        result = qa_chain.invoke(question)
        return jsonify({"answer": result.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Document Q&A Chatbot API is running."})


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
