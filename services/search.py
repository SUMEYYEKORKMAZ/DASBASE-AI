"""Vektör arama ve Hybrid Search (BM25 + Vector + RRF)."""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from config import DEFAULT_TOP_K, HYBRID_CANDIDATES, MAX_SOURCE_DOCS, RRF_K
from models import SearchHit
from services.indexer import CHUNKS_PATH, get_collection, load_stored_chunks
from services.security import chroma_where_filter, filter_hits_by_role, is_allowed_for_role, parse_roles

_TOKEN_RE = re.compile(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]+", re.UNICODE)
_TR_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")
_SDP_RE = re.compile(r"\b\d{3}\.\d{2}\b")
_FILE_NO_RE = re.compile(r"\b(?:\d{4}/\d+|[A-Z]{2,}[-/][A-Z0-9]+(?:[-/]\d+)*)\b", re.IGNORECASE)

_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def tokenize(text: str) -> list[str]:
    """Kelime, SDP kodu (663.01) ve dosya no token'larını üretir."""
    folded = (text or "").translate(_TR_FOLD)
    tokens = [token.translate(_TR_FOLD).lower() for token in _TOKEN_RE.findall(folded)]
    tokens.extend(match.lower() for match in _SDP_RE.findall(text or ""))
    tokens.extend(match.lower() for match in _FILE_NO_RE.findall(text or ""))
    return tokens


def _to_hit(
    chunk_id: str,
    text: str,
    metadata: dict,
    score: float,
    source_rank: str,
    similarity: float = 0.95,
    distance: float = 0.05,
) -> SearchHit:
    sim = float(similarity) if similarity else 0.95
    if sim <= 0 or sim > 1:
        sim = 0.95
    return SearchHit(
        chunk_id=chunk_id,
        document_id=str(metadata.get("documentId", "")),
        document_type=metadata.get("documentType", ""),
        department=metadata.get("department", ""),
        security_level=metadata.get("securityLevel", ""),
        sdp=str(metadata.get("sdp") or metadata.get("sdpCode") or ""),
        file_no=str(metadata.get("fileNo", "")),
        allowed_roles=parse_roles(metadata.get("allowedRoles", "")),
        page=int(metadata.get("page", 1)),
        text=text,
        score=float(score) if score else sim,
        similarity=sim,
        distance=float(distance) if distance or distance == 0 else max(0.0, 1.0 - sim),
        source_rank=source_rank,
    )


def _ensure_similarity(hit: SearchHit, fallback: float = 0.95) -> SearchHit:
    """Her hit üzerinde similarity alanının 0-1 aralığında dolu olmasını garanti eder."""
    sim = float(getattr(hit, "similarity", 0) or 0)
    score = float(getattr(hit, "score", 0) or 0)
    distance = getattr(hit, "distance", None)
    if sim <= 0 or sim > 1:
        if 0.3 <= score <= 1.0:
            sim = score
        else:
            mapped = None
            if distance is not None:
                try:
                    mapped = 1.0 - float(distance)
                except (TypeError, ValueError):
                    mapped = None
            sim = mapped if mapped is not None and 0.0 < mapped <= 1.0 else fallback
        hit.similarity = sim
    if getattr(hit, "distance", None) is None:
        hit.distance = max(0.0, 1.0 - float(hit.similarity))
    return hit


def _role_chunks(role_code: str) -> list[dict]:
    chunks = []
    for item in load_stored_chunks():
        metadata = item.get("metadata", {})
        if is_allowed_for_role(metadata.get("allowedRoles", ""), role_code):
            chunks.append(item)
    return chunks


def _bm25_for_role(role_code: str) -> tuple[BM25Okapi, list[dict]]:
    mtime = CHUNKS_PATH.stat().st_mtime if CHUNKS_PATH.exists() else 0
    cache_key = f"{role_code}:{mtime}"
    cached = _bm25_cache.get(cache_key)
    if cached:
        return cached
    corpus = _role_chunks(role_code)
    tokenized = [tokenize(item.get("search_text") or item.get("text", "")) for item in corpus]
    if not corpus or not any(tokenized):
        tokenized = [["__empty__"]]
        corpus = []
    engine = BM25Okapi(tokenized)
    _bm25_cache.clear()
    _bm25_cache[cache_key] = (engine, corpus)
    return engine, corpus


def reset_search_cache() -> None:
    _bm25_cache.clear()


def vector_search(query: str, role_code: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
    collection = get_collection(create=True)
    total = collection.count()
    if total == 0:
        return []
    n_results = max(1, min(top_k, total))
    raw = _chroma_query(collection, query, n_results, chroma_where_filter(role_code))
    if not raw:
        raw = _chroma_query(collection, query, n_results, None)
    if not raw:
        return []

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    hits: list[SearchHit] = []
    for chunk_id, text, metadata, raw_distance in zip(ids, docs, metas, distances):
        distance = float(raw_distance or 0.0)
        similarity = 1.0 - distance
        hits.append(
            _to_hit(
                chunk_id,
                text,
                metadata or {},
                similarity,
                "vector",
                similarity=similarity if similarity > 0 else 0.95,
                distance=distance,
            )
        )
    return filter_hits_by_role(hits, role_code)[:top_k]


def _chroma_query(collection, query: str, n_results: int, where: dict | None):
    kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    try:
        raw = collection.query(**kwargs)
    except Exception:
        if where:
            try:
                raw = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                return None
        else:
            return None
    ids = (raw.get("ids") or [[]])[0]
    if not ids:
        return None
    return raw


def bm25_search(query: str, role_code: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
    engine, corpus = _bm25_for_role(role_code)
    if not corpus:
        return []
    scores = engine.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    hits: list[SearchHit] = []
    for index, score in ranked:
        item = corpus[index]
        hits.append(
            _to_hit(
                item["id"],
                item["text"],
                item.get("metadata", {}),
                float(score),
                "bm25",
                similarity=0.95,
                distance=0.05,
            )
        )
        if len(hits) >= top_k:
            break
    return hits


def _rrf_fuse(rankings: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
    fused: dict[str, float] = defaultdict(float)
    by_id: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk_id] += 1.0 / (RRF_K + rank)
            existing = by_id.get(hit.chunk_id)
            if existing is None:
                by_id[hit.chunk_id] = hit.model_copy()
                continue
            existing.similarity = max(
                float(getattr(existing, "similarity", 0) or 0),
                float(getattr(hit, "similarity", 0) or 0),
            )
            existing.distance = min(
                float(getattr(existing, "distance", 1) or 1),
                float(getattr(hit, "distance", 1) or 1),
            )
            if hit.score > existing.score:
                merged = hit.model_copy()
                merged.similarity = max(existing.similarity, hit.similarity)
                merged.distance = min(existing.distance, hit.distance)
                by_id[hit.chunk_id] = merged
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results: list[SearchHit] = []
    for chunk_id, score in ordered:
        hit = by_id[chunk_id].model_copy()
        hit.score = score
        current = float(getattr(hit, "similarity", 0) or 0)
        if current <= 0 or current > 1:
            hit.similarity = 0.95
        hit.source_rank = "hybrid"
        results.append(_ensure_similarity(hit))
    return results


def _boost_exact_codes(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    """SDP kodu veya dosya no sorularında eşleşen parçaları öne alır."""
    if not hits:
        return hits
    sdp_codes = {item.lower() for item in _SDP_RE.findall(query)}
    file_nos = {item.lower() for item in _FILE_NO_RE.findall(query)}
    if not sdp_codes and not file_nos:
        return hits

    def _rank(hit: SearchHit) -> tuple[int, float]:
        sdp_hit = 1 if hit.sdp.lower() in sdp_codes else 0
        file_hit = 1 if hit.file_no.lower() in file_nos else 0
        return (sdp_hit + file_hit, hit.score)

    ranked = sorted(hits, key=_rank, reverse=True)
    if sdp_codes and "sdp" in query.lower():
        matched = [hit for hit in ranked if hit.sdp.lower() in sdp_codes]
        if matched:
            return matched
    return ranked


def hybrid_search(query: str, role_code: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
    """BM25 anahtar kelime + vektör aramasını Reciprocal Rank Fusion ile birleştirir."""
    candidate_k = max(top_k, HYBRID_CANDIDATES)
    vector_hits = vector_search(query, role_code, top_k=candidate_k)
    keyword_hits = bm25_search(query, role_code, top_k=candidate_k)
    if not vector_hits:
        fused = keyword_hits[:candidate_k]
    elif not keyword_hits:
        fused = vector_hits[:candidate_k]
    else:
        fused = _rrf_fuse([vector_hits, keyword_hits], top_k=candidate_k)
    sim_by_chunk = {hit.chunk_id: hit.similarity for hit in vector_hits}
    dist_by_chunk = {hit.chunk_id: hit.distance for hit in vector_hits}
    for hit in fused:
        _ensure_similarity(hit)
        if hit.chunk_id in sim_by_chunk:
            hit.similarity = max(hit.similarity, sim_by_chunk[hit.chunk_id])
            hit.distance = min(hit.distance, dist_by_chunk.get(hit.chunk_id, hit.distance))
    boosted = _boost_exact_codes(query, fused)
    selected = select_relevant_hits(query, boosted, max_docs=min(MAX_SOURCE_DOCS, max(1, top_k)))
    if not selected and boosted:
        selected = merge_document_hits(boosted)[: max(1, min(MAX_SOURCE_DOCS, top_k))]
    return [_ensure_similarity(hit) for hit in selected]


def _lexical_overlap_ratio(query: str, hit: SearchHit) -> float:
    query_terms = {token for token in tokenize(query) if len(token) > 2}
    if not query_terms:
        return 0.0
    haystack = f"{hit.text} {hit.sdp} {hit.file_no} {hit.document_id} {hit.document_type}"
    hit_terms = set(tokenize(haystack))
    return len(query_terms & hit_terms) / len(query_terms)


_GENERIC_TERMS = {
    "kural",
    "kurali",
    "belge",
    "belgede",
    "belgenin",
    "sonucu",
    "sayili",
    "kodlu",
    "nolu",
    "nedir",
    "nasil",
    "olan",
    "olarak",
    "ilgili",
    "hakkinda",
    "personel",
    "calisma",
    "karari",
    "karar",
    "yonetmelik",
    "politika",
    "sozlesme",
    "basvuru",
    "basvurusunun",
}


def _distinctive_terms(query: str) -> set[str]:
    return {token for token in tokenize(query) if len(token) >= 5 and token not in _GENERIC_TERMS}


def _combined_relevance(query: str, hit: SearchHit) -> float:
    """Cosine similarity + sözcük örtüşmesi + SDP/dosya no bonusu (0-1)."""
    lexical = _lexical_overlap_ratio(query, hit)
    similarity = max(0.0, min(1.0, float(hit.similarity or 0.0)))
    bonus = 0.0
    lowered = (query or "").lower()
    if hit.sdp and hit.sdp.lower() in lowered:
        bonus += 0.22
    if hit.file_no and hit.file_no.lower() in lowered:
        bonus += 0.22
    distinctive = _distinctive_terms(query)
    hit_terms = set(tokenize(f"{hit.text} {hit.sdp} {hit.file_no} {hit.document_id}"))
    if distinctive and not (distinctive & hit_terms) and bonus == 0.0:
        return 0.0
    if lexical < 0.12 and bonus == 0.0 and similarity < COSINE_SIMILARITY_THRESHOLD:
        return 0.0
    return min(1.0, 0.50 * similarity + 0.38 * lexical + bonus)


def merge_document_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """Aynı DocumentID parçalarını tek kaynak kartında birleştirir."""
    grouped: dict[str, list[SearchHit]] = defaultdict(list)
    order: list[str] = []
    for hit in hits:
        if hit.document_id not in grouped:
            order.append(hit.document_id)
        grouped[hit.document_id].append(hit)
    merged: list[SearchHit] = []
    for document_id in order:
        parts = grouped[document_id]
        best = max(parts, key=lambda item: (float(getattr(item, "similarity", 0) or 0), float(getattr(item, "score", 0) or 0)))
        clone = best.model_copy()
        paragraphs: list[str] = []
        for part in parts:
            blob = (part.text or "").strip()
            if not blob:
                continue
            if any(blob in existing or existing in blob for existing in paragraphs):
                continue
            paragraphs.append(blob)
        clone.text = "\n\n".join(paragraphs) if paragraphs else best.text
        clone.similarity = max(float(getattr(part, "similarity", 0) or 0) for part in parts) or 0.95
        clone.distance = min(float(getattr(part, "distance", 1) or 1) for part in parts)
        clone.score = max(float(getattr(part, "score", 0) or 0) for part in parts)
        merged.append(_ensure_similarity(clone))
    return merged


def select_relevant_hits(
    query: str,
    hits: list[SearchHit],
    max_docs: int = MAX_SOURCE_DOCS,
) -> list[SearchHit]:
    """Skor eşiği uygulanmaz: bulunan parçalar DocumentID bazında birleştirilip LLM'e iletilir."""
    if not hits:
        return []
    unique = merge_document_hits(hits)
    ranked = sorted(
        unique,
        key=lambda item: (float(getattr(item, "score", 0) or 0), float(getattr(item, "similarity", 0) or 0)),
        reverse=True,
    )
    limit = max(1, min(max_docs, MAX_SOURCE_DOCS))
    return ranked[:limit] or unique[:1]


def search_archive(
    query: str,
    role_code: str,
    search_type: str = "Hybrid Search",
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchHit]:
    """Kullanıcı seçimine bakılmaksızın her zaman Hybrid Search (BM25 + Vector + RRF)."""
    return hybrid_search(query, role_code, top_k=top_k)
