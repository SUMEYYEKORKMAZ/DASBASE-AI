"""Belge yükleme, 500 token'lık parçalama ve ChromaDB indeksleme."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    DOCUMENTS_DIR,
    KNOWN_ROLES,
    METADATA_DIR,
    STORAGE_DIR,
)
from services.embeddings import embedding_mode, get_embedding_function
from services.security import matching_role_codes, parse_roles, role_flag_key

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

CHUNKS_PATH = STORAGE_DIR / "chunks.json"
STAMP_PATH = STORAGE_DIR / "index_stamp.json"
CORPUS_PATH = DATA_DIR / "corpus.json"


def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(create: bool = True):
    client = get_chroma_client()
    kwargs: dict[str, Any] = {
        "name": COLLECTION_NAME,
        "embedding_function": get_embedding_function(),
        "metadata": {"hnsw:space": "cosine"},
    }
    try:
        if create:
            return client.get_or_create_collection(**kwargs)
        return client.get_collection(name=COLLECTION_NAME, embedding_function=get_embedding_function())
    except ValueError:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return client.get_or_create_collection(**kwargs)


def load_page_records(metadata_dir: Path | None = None) -> list[dict[str, Any]]:
    """DASBase sayfa kayıtlarını metadata JSON dosyalarından üretir."""
    source_dir = metadata_dir or METADATA_DIR
    records: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(_expand_document(payload))
    return records


def resolve_document_text(document_id: str, inline_text: str = "") -> str:
    """Metadata içindeki metni veya data/documents altındaki txt dosyasını okur."""
    if inline_text and str(inline_text).strip():
        return str(inline_text)
    if not document_id:
        return ""
    exact = DOCUMENTS_DIR / f"{document_id}.txt"
    if exact.exists():
        return exact.read_text(encoding="utf-8")
    matches = sorted(DOCUMENTS_DIR.glob(f"{document_id}*.txt"))
    if matches:
        return matches[0].read_text(encoding="utf-8")
    return ""


def _expand_document(payload: dict[str, Any]) -> list[dict[str, Any]]:
    document_id = str(payload.get("documentId", ""))
    sdp = str(payload.get("sdpCode") or payload.get("sdp") or "")
    file_no = str(payload.get("fileNo") or payload.get("file_no") or "")
    shared = {
        "documentId": document_id,
        "documentType": payload.get("documentType", ""),
        "department": payload.get("department", ""),
        "securityLevel": payload.get("securityLevel", ""),
        "sdp": sdp,
        "fileNo": file_no,
        "allowedRoles": parse_roles(payload.get("allowedRoles", [])),
    }
    pages = payload.get("pages")
    if isinstance(pages, list) and pages:
        out = []
        for page in pages:
            text = resolve_document_text(document_id, page.get("text", ""))
            if not text.strip():
                text = resolve_document_text(document_id, "")
            out.append(
                {
                    **shared,
                    "page": int(page.get("page", payload.get("page", 1))),
                    "text": text,
                }
            )
        return out
    return [
        {
            **shared,
            "page": int(payload.get("page", 1)),
            "text": resolve_document_text(document_id, payload.get("text", "")),
        }
    ]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Metni yaklaşık 500 token'lık parçalara böler."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [part.strip() for part in splitter.split_text(text or "") if part.strip()]


def build_chunks(records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    pages = records if records is not None else load_page_records()
    chunks: list[dict[str, Any]] = []
    for record in pages:
        parts = chunk_text(record.get("text", ""))
        if not parts:
            continue
        roles = parse_roles(record.get("allowedRoles", []))
        sdp = str(record.get("sdp", ""))
        file_no = str(record.get("fileNo", ""))
        for index, part in enumerate(parts, start=1):
            chunk_id = f"{record['documentId']}_p{record['page']}_c{index}"
            search_text = " ".join(
                token
                for token in (part, sdp, sdp, file_no, file_no, str(record["documentId"]))
                if token
            )
            chroma_meta = {
                "documentId": str(record["documentId"]),
                "documentType": record.get("documentType", ""),
                "department": record.get("department", ""),
                "securityLevel": record.get("securityLevel", ""),
                "sdp": sdp,
                "sdpCode": sdp,
                "fileNo": file_no,
                "allowedRoles": ",".join(roles),
                "page": int(record.get("page", 1)),
                "chunkIndex": index,
            }
            flag_roles: set[str] = set()
            for role in roles:
                flag_roles.update(matching_role_codes(role))
            for role in KNOWN_ROLES:
                chroma_meta[role_flag_key(role)] = role in flag_roles
            chunks.append(
                {
                    "id": chunk_id,
                    "text": part,
                    "search_text": search_text,
                    "metadata": chroma_meta,
                    "allowedRoles": roles,
                }
            )
    return chunks


def _stamp() -> dict[str, Any]:
    files = sorted(list(METADATA_DIR.glob("*.json")) + list(DOCUMENTS_DIR.glob("*.txt")))
    signature = "|".join(f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}" for path in files)
    return {"signature": signature, "embedding_mode": embedding_mode(), "chunk_size": CHUNK_SIZE_TOKENS}


def index_is_current() -> bool:
    if not CHUNKS_PATH.exists() or not STAMP_PATH.exists():
        return False
    try:
        saved = json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    try:
        collection = get_collection(create=True)
        if collection.count() == 0:
            return False
    except Exception:
        return False
    return saved == _stamp()


def build_index(force: bool = False) -> dict[str, Any]:
    """ChromaDB ve BM25 korpusunu oluşturur."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not force and index_is_current():
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        return {"chunk_count": len(chunks), "rebuilt": False, "embedding_mode": embedding_mode()}

    chunks = build_chunks()
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection(create=True)
    if chunks:
        collection.add(
            ids=[item["id"] for item in chunks],
            documents=[item["text"] for item in chunks],
            metadatas=[item["metadata"] for item in chunks],
        )

    serializable = [
        {
            "id": item["id"],
            "text": item["text"],
            "search_text": item.get("search_text", item["text"]),
            "metadata": item["metadata"],
        }
        for item in chunks
    ]
    CHUNKS_PATH.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    STAMP_PATH.write_text(json.dumps(_stamp(), ensure_ascii=False, indent=2), encoding="utf-8")
    export_corpus(chunks)
    try:
        from services.search import reset_search_cache

        reset_search_cache()
    except Exception:
        pass
    return {"chunk_count": len(chunks), "rebuilt": True, "embedding_mode": embedding_mode()}


def export_corpus(chunks: list[dict[str, Any]]) -> None:
    """İndekslenen parçaları DASBase şemasıyla data/corpus.json dosyasına yazar."""
    records = []
    for item in chunks:
        meta = item.get("metadata", {})
        records.append(
            {
                "documentId": str(meta.get("documentId", "")),
                "documentType": meta.get("documentType", ""),
                "department": meta.get("department", ""),
                "securityLevel": meta.get("securityLevel", ""),
                "sdp": str(meta.get("sdp") or meta.get("sdpCode") or ""),
                "sdpCode": str(meta.get("sdpCode") or meta.get("sdp") or ""),
                "fileNo": str(meta.get("fileNo", "")),
                "allowedRoles": parse_roles(meta.get("allowedRoles", "")),
                "page": int(meta.get("page", 1)),
                "text": item.get("text", ""),
            }
        )
    CORPUS_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    stats = build_index(force=True)
    print(f"İndeks hazır: {stats['chunk_count']} parça, embedding={stats['embedding_mode']}")


def load_stored_chunks() -> list[dict[str, Any]]:
    if not CHUNKS_PATH.exists():
        build_index()
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
