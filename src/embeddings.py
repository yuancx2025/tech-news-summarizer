# src/embeddings.py
"""OpenAI & Gemini embeddings wrapper for RAG pipeline."""
from __future__ import annotations

import os
from typing import Optional, Literal

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_embeddings(
    provider: Literal["openai", "gemini"] = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
):
    """Get embeddings instance. Defaults to Gemini text-embedding-004."""
    if provider == "gemini":
        model = model_name or "models/text-embedding-004"
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required for Gemini embeddings. Set the environment "
                "variable or pass api_key explicitly."
            )
        return get_gemini_embeddings(model_name=model, api_key=key)
    else:
        model = model_name or "text-embedding-3-small"
        return get_openai_embeddings(model_name=model, api_key=api_key)

def get_gemini_embeddings(model_name: str = "models/text-embedding-004",
                          api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Return a LangChain embeddings object backed by Google Gemini embeddings.

    Args:
        model_name: Gemini embeddings model. Default: "text-embedding-004".
        api_key: Optional explicit API key. If None, reads GOOGLE_API_KEY from env.

    Returns:
        GoogleGenerativeAIEmbeddings instance (compatible with LangChain vector stores).
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required for Gemini embeddings. Set the environment "
            "variable or pass api_key explicitly."
        )
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)

def get_openai_embeddings(model_name: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Thin wrapper so the rest of the code doesn't need to know implementation details.
    """
    return OpenAIEmbeddings(model=model_name)

