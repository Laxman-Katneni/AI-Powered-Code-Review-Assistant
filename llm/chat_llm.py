# llm/chat_llm.py
from typing import List, Dict, Any

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL
from llm.prompts import build_rag_prompt


def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)



# Building a RAG prompt and call the chat model
def answer_with_rag(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    client = get_client()
    prompt = build_rag_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI code assistant."},
            {"role": "user", "content": prompt},
        ],
        #temperature=0.2,
    )

    return response.choices[0].message.content.strip()
