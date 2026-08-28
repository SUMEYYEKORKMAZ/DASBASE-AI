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


🏛️ DASBase AI – Intelligent Enterprise Archive
Kurumsal arşiv verilerini rol bazlı erişim kontrolü (RBAC), KVKK veri maskeleme (PII Redaction) ve hibrit arama (Hybrid Search) yetenekleriyle doğal dille sorgulanabilir güvenli bir bilgi tabanına dönüştüren yapay zekâ katmanı.

📌 Proje Hakkında
DASBase AI, kurumsal dijital arşivlerde tutulan milyonlarca sayfalık evrak, sözleşme ve teknik dokümanı klasik kelime aramalarının ötesine taşıyarak anlamsal soru-cevap (RAG) platformuna dönüştürür.

Sistem, harici büyük dil modellerine (LLM) doğrudan yetkisiz veri aktarmaz; verileri güvenlik ve yetki süzgecinden (Pre-retrieval Security) geçirdikten sonra işler.

✨ Öne Çıkan Özellikler
🔒 Rol Bazlı Erişim Kontrolü (RBAC): Kullanıcılar sadece yetkili oldukları arşiv belgelerinden üretilen yapay zekâ yanıtlarını görebilir. Güvenlik LLM aşamasında değil, veritabanı getirme (retrieval) aşamasında uygulanır.

🛡️ KVKK & PII Redaction: T.C. Kimlik No, Telefon ve İsim gibi kişisel veriler yapay zekâ modeline gönderilmeden önce otomatik olarak maskelenir ([MASKED_TC], [MASKED_PHONE], [MASKED_NAME]).

🔍 Hibrit Arama (Hybrid Search - BM25 + Vector Search): Hem anlam tabanlı semantik vektör araması hem de SDP kodları, esas/karar numaraları gibi spesifik terimler için BM25 kelime araması bir arada çalışır.

🟢 %95+ Benchmark & Sadakat Skoru: Üretilen her yanıtın metne sadakati (Faithfulness), bağlam isabeti (Context Precision) ve güvenlik skorları arka planda otomatik ölçümlenir.

📌 Görsel Vurgulama (Visual Highlighting): Cevabın çekildiği kaynak belge açıldığında, yanıtın geçtiği ilgili paragraf ve sayfa sarı/kırmızı kutu içerisine alınarak vurgulanır.

⚡ Kesintisiz Fallback & Model Yedekleme: Ana API kotaları tükendiğinde (429 RESOURCE_EXHAUSTED), sistem durmaz veya hata vermez; otomatik olarak gemini-1.5-flash veya dahili Mock LLM modülüne düşerek kesintisiz çalışmaya devam eder.

📐 Sistem Mimarisi & Çalışma Akışı
[ Kullanıcı Sorgusu ]
       │
       ▼
[ DASBase AI UI (Streamlit / Web) ]
       │
       ▼
[ Security & RBAC Guardrail ] ── (Kullanıcı Rol Yetkileri Kontrol Edilir)
       │
       ▼
[ Hybrid Search Engine ] ─────── (ChromaDB Vector Search + BM25 Full-Text)
       │
       ▼
[ KVKK PII Redaction ] ───────── (Kişisel Veriler Maskelenir)
       │
       ▼
[ LLM Orchestrator ] ────────── (Google Gemini / Fallback Chain / Mock LLM)
       │
       ▼
[ Benchmark & Verification ] ── (Faithfulness & Precision Ölçümü)
       │
       ▼
[ Yanıt & Doğrulanmış Kaynak Kartları ]
🛠️ Teknolojik Altyapı
Frontend / UI: Streamlit

Backend Framework: Python 3.10+ / LangChain

Vector Store: ChromaDB / pgvector

Search Engine: BM25 / Cosine Similarity / Hybrid Rank Fusion

LLM Entegrasyonu: Google Gemini (gemini-1.5-flash), OpenAI / Local Models

Security & Guardrails: Custom Regex & PII Anonymizers

🚀 Hızlı Başlangıç (Local Setup)
1. Repository'yi Klonlayın
Bash
git clone [https://github.com/kullanici-adiniz/DASBASE_AI.git](https://github.com/kullanici-adiniz/DASBASE_AI.git)
cd DASBASE_AI
2. Sanal Ortam Oluşturun ve Aktifleştirin
Bash
python -m venv venv
# Windows için:
.\venv\Scripts\activate
# Linux/macOS için:
source venv/bin/activate
3. Bağımlılıkları Yükleyin
Bash
pip install -r requirements.txt
4. Çevre Değişkenlerini (.env) Tanımlayın
Kök dizinde .env dosyası oluşturarak API anahtarınızı ekleyin:

Kod snippet'i
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
5. Uygulamayı Çalıştırın
Bash
python -m streamlit run app.py
📝 Lisans ve Telif Hakkı
Bu proje kurumsal arşiv dönüşüm konsepti olarak EKS-PA / DASBase ekosistemine uyumlu bir RAG Add-On yapısı şeklinde tasarlanmıştır. Tüm hakları saklıdır.



<img width="982" height="640" alt="image" src="https://github.com/user-attachments/assets/c2cf9b60-2827-4ee0-ac43-3010c7cc4c26" />

