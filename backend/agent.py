from typing import TypedDict

from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, OPENAI_API_KEY
from rag import ask_question


class AgentState(TypedDict):
    question: str
    session_id: str
    answer: str
    sources: list


# skip retrieval for greetings/meta questions - vectorstore won't help here
DIRECT_PHRASES = [
    "summarize our conversation",
    "what did we discuss",
    "thank you",
    "thanks",
    "hello",
    "hi",
    "hey",
]


def router(state: AgentState) -> str:
    question = state["question"].strip().lower()

    if len(question.split()) < 4:
        return "direct_node"

    for phrase in DIRECT_PHRASES:
        if phrase in question:
            return "direct_node"

    return "retrieve_node"


def retrieve_node(state: AgentState) -> AgentState:
    result = ask_question(
        question=state["question"],
        session_id=state["session_id"],
    )
    return {**state, "answer": result["answer"], "sources": result["sources"]}


def direct_node(state: AgentState) -> AgentState:
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7,
    )
    response = llm.invoke(state["question"])
    return {**state, "answer": response.content, "sources": []}


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("direct_node", direct_node)

    graph.set_conditional_entry_point(
        router,
        {
            "retrieve_node": "retrieve_node",
            "direct_node": "direct_node",
        },
    )

    graph.add_edge("retrieve_node", END)
    graph.add_edge("direct_node", END)

    return graph.compile()


# compiled once at import time - don't rebuild per request
_graph = _build_graph()


def run_agent(question: str, session_id: str) -> dict:
    result = _graph.invoke({"question": question, "session_id": session_id})
    return {"answer": result["answer"], "sources": result["sources"]}
