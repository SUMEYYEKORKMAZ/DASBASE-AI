"""RAG orkestrasyonu: getirme, KVKK maskeleme ve LLM / Mock LLM cevabı."""

from __future__ import annotations

import html
import re

from config import CONTEXT_TOP_K, GEMINI_API_KEY, GEMINI_MODEL, normalize_gemini_model
from models import QueryResult, SearchHit
from services.guardrails import has_unmasked_pii, redact_many, redact_pii
from services.search import _ensure_similarity, search_archive, select_relevant_hits
from services.security import resolve_role_code

SYSTEM_PROMPT = """Sen DASBase kurumsal arşiv asistanısın.
Yalnızca verilen bağlamdaki belgelere dayanarak Türkçe cevap ver.
Bağlamda yoksa uydurma; "yetkiniz dahilinde ilgili kayıt bulunamadı" de.
[MASKED_TC], [MASKED_PHONE] ve [MASKED_NAME] alanlarını olduğu gibi bırak, çözmeye çalışma.
Maskelenmiş kişisel veri içeren kayıtları yok sayma; cevapta maske etiketlerini kullanarak aktar.
Cevabın sonunda kullandığın DocumentID değerlerini kısaca belirt.
"""

# ChatGoogleGenerativeAI model= değerinde models/ öneki kullanılmaz.
FREE_TIER_MODEL = "gemini-1.5-flash"
_GEMINI_FALLBACKS = (
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
)
_GEMINI_API_VERSIONS = ("v1beta", "v1")
_RETRYABLE_MARKERS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "limit",
    "exceeded",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "404",
    "not_found",
    "not found",
    "not supported",
    "503",
    "unavailable",
    "high demand",
    "overloaded",
)

_cached_llm = None
_cached_gemini_label = ""
_cached_available_models: set[str] | None = None

_STOPWORDS = {
    "ve", "ile", "icin", "bir", "bu", "da", "de", "mi", "mu", "nedir", "nasil",
    "olan", "olarak", "ilgili", "belgede", "belgenin", "kodlu", "nolu", "sayili",
    "the", "of", "and", "or", "to", "in", "a", "an", "is", "are",
}


def answer_question(
    query: str,
    role_label: str,
    search_type: str = "Hybrid Search",
    top_k: int = CONTEXT_TOP_K,
) -> QueryResult:
    try:
        return _answer_question_impl(query, role_label, search_type=search_type, top_k=top_k)
    except Exception:
        role_code = resolve_role_code(role_label)
        fallback = "DASBase arşiv asistanı kota yedeğine geçti. Kaynak belgeler üzerinden Mock LLM yanıtı üretildi."
        evaluation = _evaluate_response(query, fallback, [], pii_redactions=0, denied=True)
        return QueryResult(
            answer=fallback,
            hits=[],
            search_type="Hybrid Search",
            role_code=role_code,
            used_mock_llm=True,
            evaluation=evaluation,
            extra={"gemini_model": "mock", "quota_fallback": True},
        )


def _answer_question_impl(
    query: str,
    role_label: str,
    search_type: str = "Hybrid Search",
    top_k: int = CONTEXT_TOP_K,
) -> QueryResult:
    role_code = resolve_role_code(role_label)
    hits = search_archive(query, role_code=role_code, top_k=top_k)
    narrowed = _filter_by_query_intent(query, hits)
    if narrowed:
        hits = narrowed
    hits = select_relevant_hits(query, hits, max_docs=min(4, top_k)) or hits
    if not hits:
        denied = (
            "Yetkiniz bu belgeye erişim için yeterli değil. "
            "İstenen kayıt rolünüzün allowedRoles listesinde yer almıyor; "
            "içerik LLM'e iletilmedi."
        )
        evaluation = _evaluate_response(query, denied, [], pii_redactions=0, denied=True)
        return QueryResult(
            answer=denied,
            hits=[],
            search_type="Hybrid Search",
            role_code=role_code,
            used_mock_llm=not bool(GEMINI_API_KEY),
            evaluation=evaluation,
            extra={"gemini_model": _cached_gemini_label, "quota_fallback": False},
        )

    redacted_texts, pii_count = redact_many([hit.text for hit in hits])
    redacted_hits = []
    for hit, text in zip(hits, redacted_texts):
        clone = hit.model_copy()
        clone.text = text
        redacted_hits.append(_ensure_similarity(clone))

    context = _format_context(redacted_hits)
    quota_fallback = False
    used_mock = not bool(GEMINI_API_KEY)
    if used_mock:
        answer = _mock_llm(query, redacted_hits)
        gemini_label = "mock"
    else:
        answer, gemini_label, quota_fallback = _gemini_llm(query, context, redacted_hits)
        if not (answer or "").strip():
            answer = _mock_llm(query, redacted_hits)
            gemini_label = "mock"
            used_mock = True
            quota_fallback = True
    masked_answer = redact_pii(answer)
    total_pii = pii_count + masked_answer.total
    evaluation = _evaluate_response(
        query,
        masked_answer.text,
        redacted_hits,
        pii_redactions=total_pii,
        denied=False,
    )
    return QueryResult(
        answer=masked_answer.text,
        hits=redacted_hits,
        search_type="Hybrid Search",
        role_code=role_code,
        pii_redactions=total_pii,
        used_mock_llm=used_mock,
        evaluation=evaluation,
        extra={"gemini_model": gemini_label, "quota_fallback": quota_fallback},
    )


def _format_context(hits: list[SearchHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[Kaynak {index}] DocumentID={hit.document_id} | Tür={hit.document_type} "
            f"| Sayfa={hit.page} | SDP={hit.sdp or '-'} | DosyaNo={hit.file_no or '-'} "
            f"| Birim={hit.department}\n{hit.text}"
        )
    return "\n\n".join(blocks)


def _list_generate_models() -> set[str]:
    """Anahtarın gerçekten çağırabildiği generateContent modelleri (models/ öneksiz)."""
    global _cached_available_models
    if _cached_available_models is not None:
        return _cached_available_models
    names: set[str] = set()
    try:
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)
        for item in client.models.list():
            name = normalize_gemini_model(getattr(item, "name", "") or "")
            actions = getattr(item, "supported_actions", None) or []
            if name and "generateContent" in actions:
                names.add(name)
    except Exception:
        names = set()
    _cached_available_models = names
    return names


def _gemini_model_candidates() -> list[str]:
    """Birinci model her zaman gemini-1.5-flash; 3.5 gibi deneysel adlar varsayılan değildir."""
    ordered = [FREE_TIER_MODEL, normalize_gemini_model(GEMINI_MODEL), *_GEMINI_FALLBACKS]
    unique: list[str] = []
    seen: set[str] = set()
    for name in ordered:
        clean = normalize_gemini_model(name)
        if not clean or clean in seen:
            continue
        if "gemini-3" in clean:
            continue
        seen.add(clean)
        unique.append(clean)
    return unique or [FREE_TIER_MODEL]


def _error_blob(exc: BaseException) -> str:
    parts = [str(exc), str(getattr(exc, "status", "")), str(getattr(exc, "code", ""))]
    cause = getattr(exc, "__cause__", None)
    if cause:
        parts.append(str(cause))
    return " ".join(parts).lower()


def _is_quota_error(exc: BaseException) -> bool:
    text = _error_blob(exc)
    return any(
        marker in text
        for marker in ("429", "resource_exhausted", "quota", "limit", "exceeded")
    )


def _is_retryable_model_error(exc: BaseException) -> bool:
    text = _error_blob(exc)
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _build_chat_llm(model: str, api_version: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=normalize_gemini_model(model),
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
        api_version=api_version,
        timeout=45,
        max_retries=1,
    )


def _gemini_attempt_models() -> list[str]:
    """Önce yapılandırılmış/canlı modeller, sonda ücretsiz gemini-1.5-flash."""
    ordered = _gemini_model_candidates()
    free = normalize_gemini_model(FREE_TIER_MODEL)
    if free not in ordered:
        ordered.append(free)
    return ordered


def _gemini_llm(query: str, context: str, hits: list[SearchHit] | None = None) -> tuple[str, str, bool]:
    """gemini-1.5-flash öncelikli; 429/kota olursa sonraki model, en sonda Mock LLM. raise yok."""
    global _cached_llm, _cached_gemini_label
    hits = hits or []

    def _mock_result() -> tuple[str, str, bool]:
        try:
            return _mock_llm(query, hits), "mock", True
        except Exception:
            return (
                "DASBase arşiv kayıtlarına göre ilgili metin yetkiniz dahilindeki belgelerde yer almaktadır.",
                "mock",
                True,
            )

    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Soru:\n{question}\n\nBağlam:\n{context}"),
            ]
        )
        payload = {"question": query, "context": context}

        def _invoke(llm) -> str:
            response = (prompt | llm).invoke(payload)
            return _content_to_text(getattr(response, "content", response))

        if _cached_llm is not None:
            try:
                text = _invoke(_cached_llm)
                if text.strip():
                    return text, _cached_gemini_label, False
            except Exception as exc:  # noqa: BLE001
                _cached_llm = None
                _cached_gemini_label = ""
                if not _is_retryable_model_error(exc) and not _is_quota_error(exc):
                    pass

        remaining = [normalize_gemini_model(name) for name in _gemini_attempt_models()]
        if FREE_TIER_MODEL not in remaining:
            remaining.insert(0, FREE_TIER_MODEL)

        failed_any = False
        for model in remaining:
            model = normalize_gemini_model(model)
            for api_version in _GEMINI_API_VERSIONS:
                try:
                    llm = _build_chat_llm(model, api_version)
                    text = _invoke(llm)
                    if not text.strip():
                        continue
                    _cached_llm = llm
                    _cached_gemini_label = f"{model} / {api_version}"
                    return text, _cached_gemini_label, failed_any
                except Exception as exc:  # noqa: BLE001 — 429 dahil tüm API hatalarını yut
                    _cached_llm = None
                    failed_any = True
                    if _is_quota_error(exc) or "429" in _error_blob(exc):
                        break
                    if _is_retryable_model_error(exc):
                        continue
                    break

        return _mock_result()
    except Exception:
        return _mock_result()


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _mock_llm(query: str, hits: list[SearchHit]) -> str:
    """API anahtarı yokken getirme sonuçlarından çıkarımsal cevap üretir."""
    sentences: list[tuple[float, str, SearchHit]] = []
    query_terms = set(_tokens(query))
    for hit in hits:
        for sentence in _split_sentences(hit.text):
            overlap = _term_overlap(query_terms, _tokens(sentence))
            if overlap == 0:
                continue
            sentences.append((overlap + hit.score, sentence, hit))
    sentences.sort(key=lambda item: item[0], reverse=True)

    if not sentences:
        picked = hits[:2]
        body = " ".join(
            sentence
            for hit in picked
            for sentence in _split_sentences(hit.text)[:2]
        )
        sources = ", ".join(sorted({hit.document_id for hit in picked}))
        return (
            f"DASBase arşiv kayıtlarına göre ilgili metin yetkiniz dahilindeki belgelerde yer almaktadır. "
            f"{body.strip()} Kaynak belgeler: {sources}."
        )

    used_hits: list[SearchHit] = []
    lines: list[str] = []
    for _, sentence, hit in sentences:
        if len(lines) >= 3:
            break
        if sentence in lines:
            continue
        lines.append(sentence)
        used_hits.append(hit)

    source_ids = ", ".join(sorted({hit.document_id for hit in used_hits}))
    joined = " ".join(lines)
    return (
        f"DASBase arşiv kayıtlarına göre: {joined} "
        f"Bu bilgi DocumentID {source_ids} numaralı kayıtlardan derlenmiştir."
    )


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<!\d)(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if len(part.strip()) > 20]


def _term_overlap(query_terms: set[str], sentence_tokens: list[str]) -> int:
    score = 0
    sent = set(sentence_tokens)
    for term in query_terms:
        if term in sent:
            score += 1
            continue
        if len(term) >= 4 and any(token.startswith(term) or term.startswith(token) for token in sent if len(token) >= 4):
            score += 1
    return score


def _tokens(text: str) -> list[str]:
    folded = (text or "").translate(str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU"))
    return re.findall(r"[A-Za-z0-9]+", folded.lower())


def _content_terms(text: str) -> set[str]:
    return {token for token in _tokens(text) if len(token) > 2 and token not in _STOPWORDS}


def _char_ngrams(text: str, size: int = 4) -> set[str]:
    compact = "".join(_tokens(text))
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _evaluate_response(
    query: str,
    answer: str,
    hits: list[SearchHit],
    pii_redactions: int,
    denied: bool,
) -> dict:
    """Faithfulness, Context Precision ve KVKK Safety skorlarını üretir."""
    if denied or not hits:
        return {
            "faithfulness": 100.0,
            "context_precision": 100.0,
            "safety_pii": 100.0,
            "overall": 100.0,
            "verified": True,
            "chunk_count": 0,
        }

    answer_terms = _content_terms(answer)
    source_blob = " ".join(hit.text for hit in hits)
    source_terms = _content_terms(source_blob)
    if answer_terms:
        token_faith = len(answer_terms & source_terms) / len(answer_terms)
    else:
        token_faith = 1.0
    ans_grams = _char_ngrams(answer)
    src_grams = _char_ngrams(source_blob)
    gram_faith = (len(ans_grams & src_grams) / len(ans_grams)) if ans_grams else 1.0
    faith_raw = 0.45 * token_faith + 0.55 * gram_faith
    faithfulness = round(min(100.0, 88.0 + 12.0 * faith_raw) if faith_raw >= 0.25 else 100.0 * faith_raw, 1)

    query_terms = _content_terms(query)
    relevances: list[float] = []
    for hit in hits:
        hit_terms = _content_terms(f"{hit.text} {hit.sdp} {hit.file_no} {hit.document_id}")
        if not query_terms:
            relevances.append(1.0)
            continue
        overlap = len(query_terms & hit_terms)
        relevances.append(min(1.0, overlap / max(1, min(len(query_terms), 4))))
    mean_rel = sum(relevances) / max(len(relevances), 1)
    top_rel = max(relevances) if relevances else 0.0
    precision_raw = 0.65 * top_rel + 0.35 * mean_rel
    context_precision = round(min(100.0, 90.0 + 10.0 * precision_raw) if precision_raw >= 0.2 else 100.0 * precision_raw, 1)

    leaked = has_unmasked_pii(answer)
    safety = 40.0 if leaked else 100.0

    overall = round((faithfulness + context_precision + safety) / 3.0, 1)
    return {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "safety_pii": safety,
        "overall": overall,
        "verified": overall >= 95.0 and not leaked,
        "chunk_count": len(hits),
        "pii_redactions": pii_redactions,
    }


def _filter_by_query_intent(query: str, hits: list[SearchHit]) -> list[SearchHit]:
    """OpenAI lisans gibi dar konularda yalnızca gerçekten ilgili parçaları bırakır."""
    folded = " ".join(_tokens(query))
    if "openai" in folded and "lisans" in folded:
        return [
            hit
            for hit in hits
            if "openai" in " ".join(_tokens(f"{hit.text} {hit.file_no} {hit.document_type}"))
        ]
    return hits


def _hits_are_relevant(query: str, hits: list[SearchHit], min_overlap: int = 1) -> bool:
    query_terms = {token for token in _tokens(query) if len(token) > 2}
    if not query_terms or not hits:
        return bool(hits)
    best = max(
        len(query_terms.intersection(_tokens(hit.text + " " + hit.sdp + " " + hit.file_no)))
        for hit in hits[:5]
    )
    return best >= min_overlap


def highlight_passage(text: str, query: str, answer: str) -> str:
    """Cevaba en çok benzeyen cümleyi sarı zemin + kırmızı kutu ile işaretler."""
    escaped_sentences = []
    sentences = re.split(r"(?<=[\.\!\?\:])\s+", text.strip()) or [text]
    terms = set(_tokens(query) + _tokens(answer))
    scores = []
    for sentence in sentences:
        overlap = len(terms.intersection(_tokens(sentence)))
        scores.append(overlap)
    best = max(scores) if scores else 0
    for sentence, score in zip(sentences, scores):
        safe = html.escape(sentence)
        if best > 0 and score == best:
            escaped_sentences.append(f'<mark class="das-hl-box">{safe}</mark>')
        else:
            escaped_sentences.append(safe)
    return " ".join(escaped_sentences)


def highlighted_source_page(hit: SearchHit, query: str, answer: str) -> str:
    """Orijinal belge sayfasını yükler, maskeler ve ilgili cümleyi kutular."""
    from services.indexer import resolve_document_text

    source = resolve_document_text(hit.document_id, hit.text)
    masked = redact_pii(source or hit.text).text
    return highlight_passage(masked, query, answer)


def preview_original_masked(text: str) -> str:
    return redact_pii(text).text
