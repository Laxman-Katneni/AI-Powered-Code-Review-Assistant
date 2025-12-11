# llm/chat_llm.py
from typing import List, Dict, Any

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from llm.prompts import build_rag_prompt
from utils.retry import with_retry


def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

@with_retry()
def _call_chat_model(client, messages):

    if isinstance(client, OpenAI):
        return client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            # no temperature override (for new models)
        )


# Building a RAG prompt and call the chat model
def answer_with_rag(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    client = get_client()
    prompt = build_rag_prompt(question, retrieved_chunks)

    messages=[
            {"role": "system", "content": "You are a helpful AI code assistant."},
            {"role": "user", "content": prompt},
        ]

    response = _call_chat_model(client, messages)

    return response.choices[0].message.content.strip()
