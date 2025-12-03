from typing import List
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, EMBEDDING_MODEL

def get_embedding_client()-> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

# Batch embedding the list of texts
def embed_texts(texts:List[str])-> List[List[float]]:
    client = get_embedding_client()
    return client.embed_documents(texts)

def embed_query(text: str)-> List[float]:
    client = get_embedding_client()
    return client.embed_query(text)
