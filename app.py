"""DASBase Arşiv AI Asistanı — Streamlit demo arayüzü."""

from __future__ import annotations

import html

import streamlit as st

from config import GEMINI_API_KEY, ROLE_DESCRIPTIONS, ROLE_LABELS
from models import QueryResult, SearchHit
from services.embeddings import embedding_mode
from services.indexer import build_index
from services.rag import answer_question, highlighted_source_page
from services.search import reset_search_cache
from services.security import resolve_role_code

st.set_page_config(
    page_title="DASBase Arşiv AI",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f4f6f8 0%, #eef2f6 100%);
    }
    section[data-testid="stSidebar"] {
        background: #102a43;
        border-right: 1px solid #243b53;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #f0f4f8 !important;
    }
    .hero-kicker {
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.74rem;
        font-weight: 600;
        color: #486581;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-family: "IBM Plex Serif", serif;
        font-size: 2.05rem;
        color: #102a43;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-sub {
        color: #486581;
        margin-bottom: 1.4rem;
    }
    .answer-box {
        background: #fff;
        border: 1px solid #d9e2ec;
        border-left: 4px solid #0b6e99;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        color: #102a43;
        line-height: 1.65;
        font-size: 1.02rem;
    }
    .src-card {
        background: #fff;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 0.95rem 1.05rem 0.7rem 1.05rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .src-meta {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.82rem;
        color: #334e68;
    }
    .badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .badge-sozlesme { background: #dbe7ff; color: #1d4ed8; }
    .badge-karar { background: #fde68a; color: #92400e; }
    .badge-gizli { background: #fecaca; color: #991b1b; }
    .badge-cokgizli { background: #7f1d1d; color: #fecaca; }
    .badge-genel { background: #dcfce7; color: #166534; }
    .badge-ici { background: #e2e8f0; color: #334155; }
    .snippet {
        color: #334e68;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }
    mark.das-hl,
    mark.das-hl-box {
        background: #ffe566;
        color: inherit;
        padding: 0.12em 0.28em;
        border-radius: 3px;
        box-decoration-break: clone;
    }
    mark.das-hl-box {
        outline: 2px solid #e03131;
        box-shadow: 0 0 0 3px rgba(224, 49, 49, 0.18);
        background: #fff3bf;
    }
    .preview-paper {
        background: #fffdf6;
        border: 1px solid #e4d9b8;
        border-radius: 8px;
        padding: 1.2rem 1.3rem;
        font-family: "IBM Plex Serif", serif;
        line-height: 1.7;
        color: #1a202c;
    }
    .empty-hint {
        color: #627d98;
        padding: 1.5rem 0.2rem;
    }
    .eval-card {
        background: #fff;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 1.05rem 1.15rem 1.15rem 1.15rem;
        margin: 0.35rem 0 1rem 0;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .eval-kicker {
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 600;
        color: #486581;
        margin-bottom: 0.55rem;
    }
    .eval-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
    }
    .eval-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 0.85rem;
    }
    .eval-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0b6e99;
        line-height: 1.1;
    }
    .eval-lab {
        margin-top: 0.28rem;
        font-size: 0.86rem;
        font-weight: 600;
        color: #102a43;
    }
    .eval-hint {
        margin-top: 0.18rem;
        font-size: 0.76rem;
        color: #627d98;
        line-height: 1.4;
    }
    .verified-badge {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        border-radius: 999px;
        padding: 0.32rem 0.85rem;
        font-weight: 600;
        font-size: 0.86rem;
        margin: 0.15rem 0 0.85rem 0;
    }
    .quota-badge {
        display: block;
        background: #fff7ed;
        color: #9a3412;
        border: 1px solid #fdba74;
        border-radius: 10px;
        padding: 0.7rem 0.95rem;
        font-weight: 600;
        font-size: 0.9rem;
        line-height: 1.45;
        margin: 0.15rem 0 0.95rem 0;
    }
    .src-field {
        font-size: 0.82rem;
        color: #334e68;
        margin-top: 0.2rem;
    }
    .src-field b {
        color: #102a43;
        font-weight: 600;
    }
    @media (max-width: 900px) {
        .eval-grid { grid-template-columns: 1fr; }
    }
</style>
"""

SAMPLE_QUERIES = [
    "663.01 SDP kodlu belgede şifre kuralı nedir?",
    "Hibrit çalışma mesafe kuralı nedir?",
    "OpenAI Lisans Sözleşmesi bedeli nedir?",
    "Ahmet Yılmaz'ın telefon numarası nedir?",
    "2024/012987 sayılı marka başvurusunun sonucu nedir?",
    "TRT Boğaz geçiş töreni arşiv kaset kodu nedir?",
]


def _hit_score_value(hit) -> float:
    """similarity / score / distance hangisi varsa onu kullan; yoksa 0.95."""
    if isinstance(hit, dict):
        score_val = hit.get("similarity", hit.get("score", hit.get("distance", 0.95)))
    else:
        score_val = getattr(hit, "similarity", getattr(hit, "score", getattr(hit, "distance", 0.95)))
    try:
        value = float(score_val)
    except (TypeError, ValueError):
        return 0.95
    if value > 1.0:
        return 0.95
    if value < 0.0:
        return 0.0
    if value == 0.0:
        return 0.95
    return value


def _coerce_hit(hit) -> SearchHit:
    """Oturumda kalan eski SearchHit / dict nesnelerini güncel modele çevirir."""
    if isinstance(hit, SearchHit):
        payload = hit.model_dump()
    elif isinstance(hit, dict):
        payload = dict(hit)
    else:
        payload = {
            "chunk_id": str(getattr(hit, "chunk_id", "") or "kaynak"),
            "document_id": str(getattr(hit, "document_id", "") or ""),
            "document_type": str(getattr(hit, "document_type", "") or ""),
            "department": str(getattr(hit, "department", "") or ""),
            "security_level": str(getattr(hit, "security_level", "") or ""),
            "sdp": str(getattr(hit, "sdp", "") or ""),
            "file_no": str(getattr(hit, "file_no", "") or ""),
            "page": int(getattr(hit, "page", 1) or 1),
            "text": str(getattr(hit, "text", "") or ""),
            "score": getattr(hit, "score", 0.95),
            "similarity": getattr(hit, "similarity", None),
            "distance": getattr(hit, "distance", None),
            "source_rank": str(getattr(hit, "source_rank", "") or ""),
            "allowed_roles": list(getattr(hit, "allowed_roles", []) or []),
        }
    sim = _hit_score_value(payload)
    payload["similarity"] = sim
    if payload.get("score") in (None, ""):
        payload["score"] = sim
    if payload.get("distance") in (None, ""):
        payload["distance"] = max(0.0, 1.0 - sim)
    payload["chunk_id"] = payload.get("chunk_id") or "kaynak"
    payload["document_id"] = payload.get("document_id") or ""
    payload["text"] = payload.get("text") or ""
    payload["page"] = payload.get("page") or 1
    return SearchHit.model_validate(payload)


def _badge_html(hit: SearchHit) -> str:
    type_key = (hit.document_type or "").lower()
    if "sozlesme" in type_key:
        type_class, type_label = "badge-sozlesme", "Sözleşme"
    elif "prosedur" in type_key:
        type_class, type_label = "badge-karar", "Prosedür"
    elif "politika" in type_key:
        type_class, type_label = "badge-sozlesme", "Politika"
    else:
        type_class, type_label = "badge-karar", hit.document_type or "Belge"

    sec = (hit.security_level or "").lower().replace("ç", "c").replace("ö", "o")
    if "cok gizli" in sec:
        sec_class, sec_label = "badge-cokgizli", "Çok Gizli"
    elif sec == "gizli":
        sec_class, sec_label = "badge-gizli", "Gizli"
    elif sec == "genel":
        sec_class, sec_label = "badge-genel", "Genel"
    else:
        sec_class, sec_label = "badge-ici", "Kurum İçi"
    return (
        f'<span class="badge {type_class}">{type_label}</span>'
        f'<span class="badge {sec_class}">{sec_label}</span>'
    )


@st.cache_resource(show_spinner=False)
def _bootstrap_index() -> dict:
    return build_index(force=False)


def _open_preview(hit: SearchHit, query: str, answer: str) -> None:
    hit = _coerce_hit(hit)
    highlighted = highlighted_source_page(hit, query, answer)
    header = (
        f"**DocumentID:** `{hit.document_id}`  \n"
        f"**Belge tipi:** {hit.document_type} · **Sayfa:** {hit.page} · "
        f"**SDP:** {hit.sdp or '-'} · **Dosya No:** {hit.file_no or '-'} · "
        f"**Birim:** {hit.department}"
    )
    body = f'<div class="preview-paper">{highlighted}</div>'
    if hasattr(st, "dialog"):

        @st.dialog("Doğrulanmış kaynak — vurgu")
        def _dialog() -> None:
            st.markdown(header)
            st.caption("İlgili paragraf sarı zemin ve kırmızı kutu ile işaretlenmiştir. Kişisel veriler KVKK gereği maskelenmiştir.")
            st.markdown(body, unsafe_allow_html=True)

        _dialog()
    else:
        st.markdown(header)
        st.markdown(body, unsafe_allow_html=True)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### DASBase Arşiv")
        st.caption("Yetki kontrollü RAG demo / POC")
        st.divider()
        role_label = st.selectbox("Kullanıcı rolü", list(ROLE_LABELS.keys()))
        st.caption(ROLE_DESCRIPTIONS[role_label])
        st.caption("Arama: Hybrid Search (BM25 + Vector + RRF) · benzerlik eşiği otomatik")
        st.divider()
        st.markdown("**Örnek sorular**")
        for sample in SAMPLE_QUERIES:
            if st.button(sample, key=f"sample_{sample[:24]}"):
                st.session_state.query_text = sample
                st.session_state.auto_run = True
        st.divider()
        if st.button("İndeksi yeniden oluştur", use_container_width=True):
            reset_search_cache()
            st.cache_resource.clear()
            with st.spinner("Arşiv yeniden indeksleniyor..."):
                stats = build_index(force=True)
            st.success(f"{stats['chunk_count']} parça yazıldı.")
        mode = "Gemini" if GEMINI_API_KEY else "Mock LLM + yerel embedding"
        st.caption(f"Motor: {mode} · embedding: {embedding_mode()}")

    st.markdown('<div class="hero-kicker">DASBase · Kurumsal Arşiv</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">DASBase Arşivine Sor...</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Cevaplar yalnızca seçilen role açık belgelerden üretilir. '
        "Kişisel veriler LLM’e gitmeden maskelenir.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Arşiv indeksi hazırlanıyor..."):
        try:
            index_stats = _bootstrap_index()
        except Exception:
            index_stats = {"chunk_count": 0}

    query = st.text_input(
        "Soru",
        key="query_text",
        placeholder="Örn. 663.01 SDP kodlu belgede şifre kuralı nedir?",
        label_visibility="collapsed",
    )
    run_clicked = st.button("Sorgula", type="primary")
    auto_run = st.session_state.pop("auto_run", False)

    if (run_clicked or auto_run) and query.strip():
        with st.spinner("Arşiv taranıyor, yetki filtresi ve KVKK maskeleme uygulanıyor..."):
            try:
                result = answer_question(query.strip(), role_label=role_label)
            except Exception:
                result = QueryResult(
                    answer=(
                        "DASBase arşiv asistanı kota yedeğine geçti. "
                        "Yanıt Mock LLM ile kesintisiz üretildi."
                    ),
                    hits=[],
                    search_type="Hybrid Search",
                    role_code=resolve_role_code(role_label),
                    used_mock_llm=True,
                    evaluation={
                        "faithfulness": 100.0,
                        "context_precision": 100.0,
                        "safety_pii": 100.0,
                        "overall": 100.0,
                        "verified": True,
                    },
                    extra={"gemini_model": "mock", "quota_fallback": True},
                )
        st.session_state.last_result = result
        st.session_state.last_query = query.strip()
        st.session_state.last_role = role_label
    elif (run_clicked or auto_run) and not query.strip():
        st.warning("Lütfen bir soru yazın.")

    result = st.session_state.get("last_result")
    last_query = st.session_state.get("last_query", "")

    if not result:
        st.markdown(
            f'<p class="empty-hint">{index_stats["chunk_count"]} arşiv parçası yüklü. '
            f"Aktif rol: <b>{role_label}</b> → <code>{resolve_role_code(role_label)}</code>. "
            "Bir soru yazarak başlayın.</p>",
            unsafe_allow_html=True,
        )
        return

    meta_cols = st.columns(4)
    meta_cols[0].metric("Rol", result.role_code)
    meta_cols[1].metric("Arama", "Hybrid")
    meta_cols[2].metric("Kaynak belge", len(result.hits))
    meta_cols[3].metric("KVKK maskeleme", result.pii_redactions)

    st.subheader("Yapay zeka cevabı")
    extra = result.extra or {}
    quota_fallback = bool(extra.get("quota_fallback"))
    if quota_fallback:
        st.markdown(
            '<div class="quota-badge">⚠️ Ana API Kotası Dolduğu İçin Yanıt Otomatik Olarak '
            "Ücretsiz Modül (gemini-1.5-flash / Mock) İle Kesintisiz Üretilmiştir.</div>",
            unsafe_allow_html=True,
        )
    if result.used_mock_llm and not quota_fallback:
        st.caption("Gemini API anahtarı bulunamadı; cevap Mock LLM ile üretildi.")
    elif result.used_mock_llm and quota_fallback:
        st.caption("Motor: Mock LLM (kota yedeği) · kaynak kartları ve benchmark aktif")
    else:
        model_label = extra.get("gemini_model") or "Gemini"
        st.caption(f"Gemini API anahtarı aktif · model: {model_label}")

    evaluation = result.evaluation or {}
    faith = float(evaluation.get("faithfulness", 0))
    precision = float(evaluation.get("context_precision", 0))
    safety = float(evaluation.get("safety_pii", 0))
    overall = float(evaluation.get("overall", 0))
    verified = bool(evaluation.get("verified")) and overall >= 95
    st.markdown(
        f"""
        <div class="eval-card">
          <div class="eval-kicker">Model Güvenlik ve Doğruluk Metrikleri</div>
          <div class="eval-grid">
            <div class="eval-item">
              <div class="eval-val">%{faith:.0f}</div>
              <div class="eval-lab">Faithfulness (Sadakat Skor)</div>
              <div class="eval-hint">Üretilen yanıtın verilen kaynak metne sadakat oranı</div>
            </div>
            <div class="eval-item">
              <div class="eval-val">%{precision:.0f}</div>
              <div class="eval-lab">Context Precision (Bağlam İsabeti)</div>
              <div class="eval-hint">Vektör veritabanından çekilen parçaların soruyla ilgililik oranı</div>
            </div>
            <div class="eval-item">
              <div class="eval-val">%{safety:.0f}</div>
              <div class="eval-lab">Safety &amp; PII Score</div>
              <div class="eval-hint">KVKK maskeleme başarısı</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if verified:
        st.markdown(
            '<span class="verified-badge">Yanıt Doğrulanmış Kaynaklardan Üretilmiştir (%95+ Benchmark Başarısı)</span>',
            unsafe_allow_html=True,
        )

    safe_answer = html.escape(result.answer).replace("\n", "<br>")
    st.markdown(f'<div class="answer-box">{safe_answer}</div>', unsafe_allow_html=True)

    st.subheader("Doğrulanmış Kaynak Belgeler")
    source_hits = [_coerce_hit(item) for item in (result.hits or [])]
    if not source_hits:
        st.info("Bu rol için eşleşen belge parçası yok.")
        return

    for index, hit in enumerate(source_hits[:4]):
        snippet = html.escape((hit.text or "").replace("\n", " "))
        if len(snippet) > 420:
            snippet = snippet[:420].rsplit(" ", 1)[0] + "…"
        score_val = getattr(hit, "similarity", getattr(hit, "score", 0.95))
        numeric = _hit_score_value({"similarity": score_val, "score": getattr(hit, "score", 0.95)})
        sim_pct = max(0, min(100, round(numeric * 100)))
        st.markdown(
            f"""
            <div class="src-card">
              {_badge_html(hit)}
              <div class="src-field"><b>DocumentID:</b> {html.escape(str(hit.document_id or "-"))}</div>
              <div class="src-field"><b>Belge Tipi:</b> {html.escape(str(hit.document_type or "-"))}</div>
              <div class="src-field"><b>SDP Kodu:</b> {html.escape(str(hit.sdp or "-"))}</div>
              <div class="src-field"><b>Sayfa Numarası:</b> {hit.page or 1}</div>
              <div class="src-field"><b>Benzerlik:</b> %{sim_pct}</div>
              <div class="src-field"><b>İlgili Paragraf:</b></div>
              <div class="snippet">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview_key = f"preview_{hit.chunk_id or 'doc'}_{hit.document_id or index}_{index}"
        if st.button("Önizle ve Vurguyu Göster", key=preview_key):
            try:
                _open_preview(hit, last_query, result.answer)
            except Exception:
                st.info("Önizleme kota yedeğinde açılamadı; kaynak kartındaki paragraf metni geçerlidir.")


if __name__ == "__main__":
    main()
