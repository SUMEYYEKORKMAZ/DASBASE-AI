# DASBase Arşiv AI Asistanı (Demo / POC)

Yetki kontrollü RAG, hybrid search (BM25 + Chroma + RRF) ve KVKK maskeleme içeren kurumsal arşiv demosu.

## Mimari

```
Streamlit :8501  ──┐
FastAPI  /query  ──┤
                   ▼
 services/security.py   RBAC metadata filter (allowedRoles)
 services/search.py     Vector + BM25 + RRF Hybrid Search
 services/guardrails.py [MASKED_TC] / [MASKED_PHONE] / [MASKED_NAME]
 services/rag.py        LangChain Gemini (gemini-1.5-flash) veya Mock LLM
                   ▼
        ChromaDB (lokal) + data/corpus.json
```

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
copy .env.example .env
python -m services.indexer
streamlit run app.py
```

İsteğe bağlı API: `uvicorn api:app --reload --port 8000`

`GEMINI_API_KEY` yoksa Mock LLM + yerel hashing embedding kullanılır.

## Test senaryoları

1. **Nokta atışı SDP:** `663.01 SDP kodlu belgede şifre kuralı nedir?` → Hybrid Search, şifre en az 12 karakter.
2. **RBAC duvarı:** Satınalma Uzmanı ile `OpenAI Lisans Sözleşmesi bedeli nedir?` → erişim engeli. Genel Müdür ile aynı soru → yıllık $250,000.
3. **KVKK:** `Ahmet Yılmaz'ın telefon numarası nedir?` → `[MASKED_PHONE]` / `[MASKED_TC]` / `[MASKED_NAME]`.

## Roller

Satınalma Uzmanı, IK Personeli, Hukuk Müşaviri, Patent Uzmanı, TRT Arşivci, Genel Müdür.

## Teknoloji

- UI: Streamlit · API: FastAPI
- RAG: LangChain (`gemini-1.5-flash` / `models/embedding-001`)
- Vektör DB: ChromaDB · Anahtar kelime: Rank-BM25 · Birleştirme: RRF
