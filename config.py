# config.py
"""
Projenin tüm ayarları tek noktada.

Her ayarın bir varsayılan değeri vardır; istenirse aynı isimli (RAG_ önekli)
ortam değişkeniyle, koda dokunmadan değiştirilebilir. Örnek:

    RAG_LLM_MODEL=mistral RAG_RETRIEVER_K=6 streamlit run app.py
"""
import os

PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))


def _ortamdan_oku(isim, varsayilan, tip=str):
    """
    Ortam değişkeni tanımlı ve boş değilse onu 'tip'e çevirerek, değilse
    varsayılanı döndürür. Geçersiz bir değer (örn. RAG_RETRIEVER_K=abc)
    sessizce yutulmaz, anlaşılır bir hatayla bildirilir.
    """
    deger = os.getenv(isim)
    if deger is None or deger.strip() == "":
        return varsayilan
    try:
        return tip(deger.strip())
    except ValueError as e:
        raise ValueError(
            f"'{isim}' ortam değişkeni geçersiz: {deger!r} ({tip.__name__} bekleniyordu)"
        ) from e


def _dizin(isim, varsayilan_klasor):
    """Göreli yollar, komutun çalıştırıldığı klasöre değil proje klasörüne göre çözülür."""
    yol = _ortamdan_oku(isim, varsayilan_klasor)
    return yol if os.path.isabs(yol) else os.path.join(PROJE_DIZINI, yol)


# Model Ayarları
EMBED_MODEL = _ortamdan_oku("RAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = _ortamdan_oku("RAG_LLM_MODEL", "llama3")

# Dizin Ayarları
DB_DIR = _dizin("RAG_DB_DIR", "chroma_db")                   # CLI'nin veritabanı
WEB_DB_DIR = _dizin("RAG_WEB_DB_DIR", "web_chroma_db")       # Web arayüzünün veritabanı (bilerek ayrı)
UPLOAD_KLASORU = _dizin("RAG_UPLOAD_KLASORU", "yuklenen_pdfler")

# Parçalama (Chunking) Ayarları
CHUNK_SIZE = _ortamdan_oku("RAG_CHUNK_SIZE", 1000, int)
CHUNK_OVERLAP = _ortamdan_oku("RAG_CHUNK_OVERLAP", 200, int)

# RAG Ayarları
RETRIEVER_K = _ortamdan_oku("RAG_RETRIEVER_K", 4, int)  # Aramada getirilecek parça sayısı (web'de kaydırıcının varsayılanı)
NUM_CTX = _ortamdan_oku("RAG_NUM_CTX", 8192, int)       # Ollama modelinin bağlam penceresi (token). Yüksek k değerlerinde büyütün.
GECMIS_MAX_MESAJ = _ortamdan_oku("RAG_GECMIS_MAX_MESAJ", 10, int)  # Modele gönderilecek en fazla geçmiş mesaj (soru+cevap = 2 mesaj)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP, CHUNK_SIZE'dan küçük olmalıdır.")
if RETRIEVER_K < 1:
    raise ValueError("RETRIEVER_K en az 1 olmalıdır.")
if GECMIS_MAX_MESAJ < 0:
    raise ValueError("GECMIS_MAX_MESAJ negatif olamaz.")
