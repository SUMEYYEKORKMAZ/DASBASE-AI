"""DASBase AI Demo yapılandırması."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
METADATA_DIR = DATA_DIR / "metadata"
CHROMA_DIR = ROOT_DIR / "chroma_db"
STORAGE_DIR = ROOT_DIR / "storage"

COLLECTION_NAME = "dasbase_archive"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
DEFAULT_TOP_K = 4
CONTEXT_TOP_K = 4
MAX_SOURCE_DOCS = 4
RRF_K = 60
HYBRID_CANDIDATES = 16
# Chroma cosine distance = 1 - cosine similarity. Eşik: similarity >= 0.38 (distance <= 0.62)
COSINE_SIMILARITY_THRESHOLD = 0.38
COSINE_DISTANCE_MAX = 0.62
VECTOR_PREFILTER_MIN_SIM = 0.12

GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def normalize_gemini_model(name: str) -> str:
    """v1 / v1beta uyumu için models/ önekini kaldırır."""
    raw = (name or "gemini-1.5-flash").strip()
    if raw.startswith("models/"):
        raw = raw[len("models/") :]
    return raw


def resolve_embedding_model(name: str) -> str:
    """Eski embedding-001 adını güncel Gemini embedding modeline çevirir."""
    raw = normalize_gemini_model(name or GEMINI_EMBEDDING_MODEL)
    if raw in {"embedding-001", "text-embedding-004"}:
        return "gemini-embedding-001"
    return raw or "gemini-embedding-001"

ROLE_LABELS = {
    "Satınalma Uzmanı": "Satinalma_Uzmani",
    "IK Personeli": "IK_Personeli",
    "Hukuk Müşaviri": "Hukuk_Musaviri",
    "Patent Uzmanı": "Patent_Uzmani",
    "TRT Arşivci": "TRT_Arsivci",
    "Genel Müdür": "Genel_Mudur",
}

ROLE_EQUIVALENTS = {
    "Genel_Mudur": ("Genel_Mudur", "Yonetici"),
    "Yonetici": ("Genel_Mudur", "Yonetici"),
}

KNOWN_ROLES = (
    "Satinalma_Uzmani",
    "IK_Personeli",
    "Hukuk_Musaviri",
    "Patent_Uzmani",
    "TRT_Arsivci",
    "Editor",
    "Genel_Mudur",
    "Yonetici",
)

ROLE_DESCRIPTIONS = {
    "Satınalma Uzmanı": "Satınalma ve ihale dosyalarına erişir; çok gizli lisans ve İK sicilini görmez.",
    "IK Personeli": "İnsan kaynakları, izin ve hibrit çalışma kararlarını görür.",
    "Hukuk Müşaviri": "İhale, lisans, marka kararı ve hukuk arşivine erişir.",
    "Patent Uzmanı": "Marka/patent kararlarını görür; satınalma ve İK özlüğünü görmez.",
    "TRT Arşivci": "Yayın transkripti ve TRT arşiv kayıtlarına erişir.",
    "Genel Müdür": "Tüm güvenlik seviyelerindeki arşiv belgelerine erişir.",
}
