"""Embedding üreticileri: Gemini veya indirmesiz lokal hashing."""

from __future__ import annotations

import hashlib
import math
import re

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, resolve_embedding_model

_TOKEN_RE = re.compile(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+")
LOCAL_EMBED_DIM = 384


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Karakter n-gram hashing. Model indirme gerektirmez, demo için yeterlidir."""

    def __init__(self, dim: int = LOCAL_EMBED_DIM) -> None:
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [_hash_embed(text, self.dim) for text in input]

    @staticmethod
    def name() -> str:
        return "dasbase_hashing"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction(dim=int(config.get("dim", LOCAL_EMBED_DIM)))


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """LangChain GoogleGenerativeAIEmbeddings sarmalayıcısı (ChromaDB uyumlu)."""

    def __init__(self, api_key: str, model: str = GEMINI_EMBEDDING_MODEL) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = resolve_embedding_model(model)
        self._embedder = GoogleGenerativeAIEmbeddings(
            model=self.model,
            google_api_key=api_key,
        )

    def __call__(self, input: Documents) -> Embeddings:
        texts = [text if isinstance(text, str) else str(text) for text in input]
        return self._embedder.embed_documents(texts)

    @staticmethod
    def name() -> str:
        return "gemini_embedding_001"

    def get_config(self) -> dict:
        return {"model": self.model}

    @staticmethod
    def build_from_config(config: dict) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model=str(config.get("model", GEMINI_EMBEDDING_MODEL)),
        )


def _hash_embed(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    tokens = _TOKEN_RE.findall((text or "").lower())
    grams: list[str] = []
    for token in tokens:
        grams.append(token)
        padded = f"#{token}#"
        for size in (2, 3):
            for index in range(len(padded) - size + 1):
                grams.append(padded[index : index + size])
    for gram in grams:
        digest = hashlib.md5(gram.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embedding_mode() -> str:
    return "gemini" if GEMINI_API_KEY else "local-hash"


def get_embedding_function():
    if GEMINI_API_KEY:
        return GeminiEmbeddingFunction(api_key=GEMINI_API_KEY, model=GEMINI_EMBEDDING_MODEL)
    return HashingEmbeddingFunction()
