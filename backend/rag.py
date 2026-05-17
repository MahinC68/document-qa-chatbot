from __future__ import annotations

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from config import VECTORSTORE_FOLDER, LLM_MODEL, OPENAI_API_KEY, get_embeddings_model

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant that answers questions based on the provided documents. "
        "Always be concise and accurate. If the answer is not in the documents, say so clearly.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# keyed by session_id so each user keeps their own conversation history
session_memory: dict[str, ConversationBufferMemory] = {}


def get_retriever():
    """Load the persisted FAISS index and return an MMR retriever.

    MMR (Maximal Marginal Relevance) diversifies results so the chain gets
    broader context rather than k near-duplicate chunks.
    """
    if not os.path.exists(VECTORSTORE_FOLDER):
        raise FileNotFoundError(
            "No vectorstore found — upload a document before asking questions."
        )

    embeddings = get_embeddings_model()
    vectorstore = FAISS.load_local(
        VECTORSTORE_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10},
    )


def get_memory(session_id: str) -> ConversationBufferMemory:
    """Return existing memory for a session, or create a fresh one.

    memory_key must match the chain's expected input key so chat history
    is injected into the prompt automatically.
    """
    if session_id not in session_memory:
        session_memory[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
    return session_memory[session_id]


def ask_question(question: str, session_id: str) -> dict:
    """Run a question through the ConversationalRetrievalChain and return the answer with sources."""
    retriever = get_retriever()
    memory = get_memory(session_id)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

    # test the retriever directly before handing off to the chain
    direct_hits = retriever.invoke(question)
    print(f"[rag] retriever direct test: {len(direct_hits)} doc(s) for {question!r}")
    for i, doc in enumerate(direct_hits):
        print(f"[rag] hit[{i}]: meta={doc.metadata} | {doc.page_content[:150]!r}")

    result = chain.invoke({"question": question})

    retrieved = result.get("source_documents", [])
    print(f"[rag] {len(retrieved)} chunk(s) retrieved for: {question!r}")
    for i, doc in enumerate(retrieved):
        print(f"[rag] chunk[{i}]: meta={doc.metadata}, snippet={doc.page_content[:120]!r}")

    # Deduplicate sources — multiple chunks can come from the same page
    seen = set()
    sources = []
    for doc in result.get("source_documents", []):
        key = (doc.metadata.get("filename"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"document": key[0], "page": key[1]})

    return {"answer": result["answer"], "sources": sources}
