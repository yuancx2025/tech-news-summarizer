# src/embeddings.py
from langchain_openai import OpenAIEmbeddings

def get_openai_embeddings(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Thin wrapper so the rest of the code doesn't need to know implementation details.
    """
    return OpenAIEmbeddings(model=model_name)