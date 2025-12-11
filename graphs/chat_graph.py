# graphs/chat_graph.py
from __future__ import annotations

from typing import List

from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from retrieval.retriever import retrieve_chunks


def build_chat_graph(repo_id: str):
    """
    Build a LangGraph workflow for chatting about a specific repo.
    Nodes:
      - retrieve: add relevant code context based on latest user question
      - llm: answer using ChatOpenAI with the full message history
    """

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,  # more deterministic answers
    )

    def retrieve_node(state: MessagesState) -> MessagesState:
        """
        Look at the most recent human message, retrieve relevant code chunks,
        and append a system message with that context.
        """
        messages: List[BaseMessage] = state["messages"]

        # Find last human message
        last_user = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user = m
                break

        if last_user is None:
            return state

        question = last_user.content

        try:
            retrieved = retrieve_chunks(repo_id, question, k=6)
        except Exception:
            retrieved = []

        if not retrieved:
            # No context found; proceed with just the chat history.
            return state

        # Build a context block from retrieved chunks
        context_parts = []
        for r in retrieved:
            meta = r["metadata"]
            path = meta["file_path"]
            start = meta["start_line"]
            end = meta["end_line"]
            context_parts.append(
                f"File: {path} (lines {start}-{end})\n{r['content']}"
            )

        context_text = "\n\n".join(context_parts)

        messages.append(
            SystemMessage(
                content=(
                    "Here is relevant context from the repository for the user's question:\n\n"
                    f"{context_text}"
                )
            )
        )

        state["messages"] = messages
        return state

    def llm_node(state: MessagesState) -> MessagesState:
        """
        Call the chat model with the accumulated messages (history + context)
        and append the assistant's reply.
        """
        messages: List[BaseMessage] = state["messages"]
        response: AIMessage = llm.invoke(messages)
        messages.append(response)
        state["messages"] = messages
        return state

    g = StateGraph(MessagesState)

    g.add_node("retrieve", retrieve_node)
    g.add_node("llm", llm_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "llm")
    g.add_edge("llm", END)

    return g.compile()
