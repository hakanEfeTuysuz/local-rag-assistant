"""
Web arayüzü (app.py) ve CLI (soru_cevap.py, rag_motoru.py) tarafından ORTAK
kullanılan RAG çekirdeği: modellerin yüklenmesi, veritabanının açılması,
promptlar, zincirin kurulması, sohbet geçmişinin kırpılması ve kaynak listesi.

Bu mantık daha önce app.py ve soru_cevap.py içinde ayrı ayrı kopyalanmıştı ve
zamanla birbirinden ayrışmıştı (örn. CLI'da konu uyumu kontrolü ve num_ctx
yoktu). Artık tek bir yerde duruyor; iki arayüz de aynı davranışı gösterir.
"""
import logging
import os

from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import EMBED_MODEL, GECMIS_MAX_MESAJ, LLM_MODEL, NUM_CTX


# ---------------------------------------------------------------------------
# Hata sınıfları: genel 'Exception' yerine, hatanın hangi aşamada olduğunu
# belli eden sınıflar. Hepsi RagHatasi'ndan türediği için tek seferde de
# yakalanabilir.
# ---------------------------------------------------------------------------
class RagHatasi(Exception):
    """Projeye ait tüm hataların ortak üst sınıfı."""


class PdfOkumaHatasi(RagHatasi):
    """PDF okunamadığında veya içinden metin çıkarılamadığında."""


class VeritabaniHatasi(RagHatasi):
    """Vektör veritabanı açılamadığında, bulunamadığında veya yazılamadığında."""


class ModelHatasi(RagHatasi):
    """Ollama'ya ulaşılamadığında veya model bulunamadığında."""


def cli_loglamasini_ayarla() -> None:
    """CLI'da aşama mesajlarını gösterir; kütüphanelerin HTTP isteği loglarını gizler."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for gurultulu in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(gurultulu).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Promptlar
# ---------------------------------------------------------------------------
SISTEM_PROMPTU = (
    "Sen, kullanıcının yüklediği belgeler üzerinden bilgi çıkaran, yerel donanımda çalışan zeki bir yapay zeka asistanısın. "
    "Sana verilen bağlam (context) İngilizce olsa bile, "
    "SEN HER ZAMAN SADECE TÜRKÇE cevap vereceksin. "
    "Cevabında tek bir İngilizce kelime veya cümle bile kullanma; "
    "bağlamdaki bilgileri kendi cümlelerinle Türkçeye çevirerek anlat. "
    "Sana verilen bağlam (context) bilgilerini kullanarak kullanıcının sorusunu detaylı ve somut şekilde cevapla; "
    "yüzeysel/genel geçmeden, bağlamdaki spesifik bilgileri (tanımlar, örnekler, sayılar) kullan. "
    "ÇOK ÖNEMLİ: Sana verilen bağlam parçaları, kullanıcının sorduğu KONUYLA uyuşmuyorsa "
    "(örneğin kullanıcı bir konu sordu ama bağlamda tamamen farklı bir konu/örnek/kod var), "
    "bunu asla doğruymuş gibi sunma. Bunun yerine açıkça şunu söyle: "
    "'Verilen bağlamda bu konuya dair [bağlamda gerçekte ne olduğunu belirt] bulunuyor, "
    "ancak sorduğunuz [X] konusuna dair doğrudan bir içerik bulamadım.' "
    "Bağlamdaki içerik ile soru arasındaki konu uyumunu her cevaptan önce kontrol et. "
    "Önceki konuşma geçmişini de dikkate alarak tutarlı cevaplar ver. "
    "Eğer cevabı bağlamda hiç bulamazsan, uydurma, sadece bilmediğini söyle.\n\n"
    "Bağlam:\n{context}"
)

# Sohbet geçmişi varken, retriever'a gidecek sorguyu geçmişten bağımsız hale getirir.
# Örn. "Onun okuduğu üniversite neresi?" -> "Hakan Efe Tüysüz'ün okuduğu üniversite neresi?"
SORU_YENIDEN_YAZMA_PROMPTU = (
    "Aşağıda bir sohbet geçmişi ve kullanıcının son sorusu var. "
    "Son soru, geçmişteki bir kişiye, nesneye veya konuya atıfta bulunuyor olabilir "
    "(örneğin 'onun', 'bunun', 'o proje', 'peki ya ikincisi'). "
    "Görevin: son soruyu, sohbet geçmişini görmeyen birinin de anlayabileceği, "
    "kendi başına anlamlı TEK bir soru olarak yeniden yazmak. "
    "Soruyu CEVAPLAMA. Yeniden yazmaya gerek yoksa soruyu olduğu gibi döndür. "
    "Sadece soruyu yaz; açıklama, tırnak veya ek metin ekleme. Soruyu Türkçe yaz."
)


# ---------------------------------------------------------------------------
# Modeller ve veritabanı
# ---------------------------------------------------------------------------
def _model_hatasi(model: str, e: Exception) -> ModelHatasi:
    return ModelHatasi(
        f"'{model}' modeli yüklenemedi. Ollama çalışıyor mu ('ollama serve') ve model indirildi mi "
        f"('ollama pull {model}')? Detay: {e}"
    )


def embedding_yukle(embed_model: str = EMBED_MODEL):
    """
    Embedding modelini hazırlar. validate_model_on_init sayesinde Ollama
    kapalıysa veya model indirilmemişse hata ilk kullanımda değil, burada,
    anlaşılır bir mesajla ortaya çıkar.
    """
    from langchain_ollama import OllamaEmbeddings

    try:
        return OllamaEmbeddings(model=embed_model, validate_model_on_init=True)
    except Exception as e:
        raise _model_hatasi(embed_model, e) from e


def llm_yukle(llm_model: str = LLM_MODEL, num_ctx: int = NUM_CTX):
    """Sohbet modelini (bağlam penceresi ayarlı) hazırlar ve varlığını doğrular."""
    from langchain_ollama import ChatOllama

    try:
        return ChatOllama(model=llm_model, num_ctx=num_ctx, validate_model_on_init=True)
    except Exception as e:
        raise _model_hatasi(llm_model, e) from e


def modelleri_yukle():
    """Embedding ve sohbet modelini birlikte döndürür: (embeddings, llm)."""
    return embedding_yukle(), llm_yukle()


def vektor_db_ac(persist_dir: str, embeddings) -> Chroma:
    """Verilen klasördeki kalıcı Chroma veritabanını açar (yoksa oluşturur)."""
    try:
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    except Exception as e:
        raise VeritabaniHatasi(f"Vektör veritabanı açılamadı ('{persist_dir}'). Detay: {e}") from e


def veritabani_bos_mu(vektor_db) -> bool:
    """Veritabanında hiç parça yoksa True döner (sadece id'ler istenir, hızlıdır)."""
    try:
        return len(vektor_db.get(limit=1, include=[]).get("ids", [])) == 0
    except Exception as e:
        raise VeritabaniHatasi(f"Vektör veritabanı okunamadı. Detay: {e}") from e


# ---------------------------------------------------------------------------
# Zincir
# ---------------------------------------------------------------------------
def zinciri_kur(llm, retriever):
    """
    Tam RAG zincirini kurar:
      1) Sohbet geçmişi varsa soru önce geçmişten bağımsız hale getirilir
         (query rewriting), retriever bu yeniden yazılmış soruyla arama yapar.
         Geçmiş boşsa bu adım atlanır, ek bir model çağrısı yapılmaz.
      2) Bulunan parçalar + geçmiş + soru, cevap üretimi için modele gider.
    Dönen zincir {"input", "chat_history"} alır; {"answer", "context", ...} döndürür.
    """
    yeniden_yazma_promptu = ChatPromptTemplate.from_messages([
        ("system", SORU_YENIDEN_YAZMA_PROMPTU),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    gecmise_duyarli_retriever = create_history_aware_retriever(llm, retriever, yeniden_yazma_promptu)

    cevap_promptu = ChatPromptTemplate.from_messages([
        ("system", SISTEM_PROMPTU),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    soru_cevap_zinciri = create_stuff_documents_chain(llm, cevap_promptu)
    return create_retrieval_chain(gecmise_duyarli_retriever, soru_cevap_zinciri)


def gecmisi_kirp(chat_history: list[BaseMessage], max_mesaj: int = GECMIS_MAX_MESAJ) -> list[BaseMessage]:
    """
    Modele gönderilecek geçmişi son 'max_mesaj' mesajla sınırlar; böylece uzun
    sohbetlerde bağlam penceresi (num_ctx) taşıp prompt sessizce kesilmez.
    Kırpılmış geçmiş her zaman bir kullanıcı (Human) mesajıyla başlar, yarım
    kalmış bir soru-cevap çifti modele gönderilmez. Orijinal liste değişmez.
    """
    if max_mesaj <= 0:
        return []
    kirpilmis = list(chat_history[-max_mesaj:])
    while kirpilmis and not isinstance(kirpilmis[0], HumanMessage):
        kirpilmis.pop(0)
    return kirpilmis


def kaynaklari_listele(dokumanlar) -> list[str]:
    """
    Retriever'ın getirdiği doküman parçalarından benzersiz (dosya, sayfa)
    kaynak listesi çıkarır. PDF sayfaları 0'dan sayılır,
    kullanıcıya göstermeden önce 1 ekliyoruz.
    """
    kaynaklar = []
    for doc in dokumanlar:
        dosya_adi = os.path.basename(doc.metadata.get("source", "bilinmeyen dosya"))
        sayfa = doc.metadata.get("page")
        if sayfa is not None:
            kaynaklar.append(f"{dosya_adi}, sayfa {sayfa + 1}")
        else:
            kaynaklar.append(dosya_adi)
    return list(dict.fromkeys(kaynaklar))
