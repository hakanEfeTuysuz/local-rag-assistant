from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever

from rag_zinciri import gecmisi_kirp, kaynaklari_listele, veritabani_bos_mu, zinciri_kur


class SahteDoc:
    """
    kaynaklari_listele fonksiyonu sadece .metadata attribute'una bakıyor,
    gerçek bir langchain Document nesnesi kurmaya gerek yok.
    """
    def __init__(self, source, page):
        self.metadata = {"source": source, "page": page}


class KaydedenRetriever(BaseRetriever):
    """Kendisine gelen arama sorgularını kaydeden, sabit doküman döndüren retriever."""
    sorgular: list = []
    dokumanlar: list = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        self.sorgular.append(query)
        return self.dokumanlar


# --- kaynaklari_listele -----------------------------------------------------

def test_tek_dosya_tek_sayfa():
    dokumanlar = [SahteDoc("ornek.pdf", 0)]
    assert kaynaklari_listele(dokumanlar) == ["ornek.pdf, sayfa 1"]


def test_sayfa_numarasi_birle_basliyor():
    # PDF sayfaları 0'dan sayılır, kullanıcıya 1'den başlayarak gösterilmeli
    dokumanlar = [SahteDoc("cv.pdf", 4)]
    assert kaynaklari_listele(dokumanlar) == ["cv.pdf, sayfa 5"]


def test_tekrarlanan_kaynaklar_temizlenir():
    dokumanlar = [
        SahteDoc("cv.pdf", 0),
        SahteDoc("cv.pdf", 0),   # aynı dosya, aynı sayfa - tekrar
        SahteDoc("rapor.pdf", 2),
    ]
    assert kaynaklari_listele(dokumanlar) == ["cv.pdf, sayfa 1", "rapor.pdf, sayfa 3"]


def test_farkli_klasordeki_ayni_isimli_dosya():
    # os.path.basename kullanıldığı için tam yol farklı olsa da sadece dosya adı gösterilir
    dokumanlar = [SahteDoc("/home/kullanici/belgeler/cv.pdf", 0)]
    assert kaynaklari_listele(dokumanlar) == ["cv.pdf, sayfa 1"]


def test_sayfa_bilgisi_yoksa_sadece_dosya_adi():
    doc = SahteDoc("notlar.pdf", None)
    assert kaynaklari_listele([doc]) == ["notlar.pdf"]


def test_bos_liste():
    assert kaynaklari_listele([]) == []


# --- gecmisi_kirp -----------------------------------------------------------

def _gecmis(tur_sayisi):
    mesajlar = []
    for i in range(tur_sayisi):
        mesajlar += [HumanMessage(content=f"soru {i}"), AIMessage(content=f"cevap {i}")]
    return mesajlar


def test_kisa_gecmis_aynen_kalir():
    gecmis = _gecmis(2)
    assert gecmisi_kirp(gecmis, max_mesaj=10) == gecmis


def test_uzun_gecmis_son_mesajlara_kirpilir():
    kirpilmis = gecmisi_kirp(_gecmis(10), max_mesaj=4)
    assert [m.content for m in kirpilmis] == ["soru 8", "cevap 8", "soru 9", "cevap 9"]


def test_kirpilmis_gecmis_her_zaman_soruyla_baslar():
    kirpilmis = gecmisi_kirp(_gecmis(5), max_mesaj=3)  # tek sayı: yarım çift atılmalı
    assert isinstance(kirpilmis[0], HumanMessage)
    assert [m.content for m in kirpilmis] == ["soru 4", "cevap 4"]


def test_sifir_limit_bos_gecmis_dondurur():
    assert gecmisi_kirp(_gecmis(3), max_mesaj=0) == []


def test_orijinal_gecmis_degismez():
    gecmis = _gecmis(6)
    gecmisi_kirp(gecmis, max_mesaj=2)
    assert len(gecmis) == 12


# --- zinciri_kur (sahte LLM ile, Ollama'sız) --------------------------------

def test_ilk_soruda_soru_aynen_aranir_ve_kaynak_doner():
    retriever = KaydedenRetriever(
        sorgular=[], dokumanlar=[Document(page_content="Hakan GTÜ'de okuyor.", metadata={"source": "cv.pdf", "page": 0})]
    )
    llm = FakeListChatModel(responses=["Gebze Teknik Üniversitesi."])
    zincir = zinciri_kur(llm, retriever)

    sonuc = zincir.invoke({"input": "Hakan nerede okuyor?", "chat_history": []})

    assert retriever.sorgular == ["Hakan nerede okuyor?"]  # geçmiş yokken yeniden yazma yapılmaz
    assert sonuc["answer"] == "Gebze Teknik Üniversitesi."
    assert kaynaklari_listele(sonuc["context"]) == ["cv.pdf, sayfa 1"]


def test_gecmis_varsa_soru_yeniden_yazilip_oyle_aranir():
    retriever = KaydedenRetriever(sorgular=[], dokumanlar=[])
    # 1. çağrı: soru yeniden yazma, 2. çağrı: cevap üretimi
    llm = FakeListChatModel(responses=["Hakan'ın okuduğu üniversite neresi?", "Bilmiyorum."])
    zincir = zinciri_kur(llm, retriever)

    gecmis = [HumanMessage(content="Bu CV kimin?"), AIMessage(content="Hakan Efe Tüysüz'ün.")]
    sonuc = zincir.invoke({"input": "Onun üniversitesi neresi?", "chat_history": gecmis})

    assert retriever.sorgular == ["Hakan'ın okuduğu üniversite neresi?"]
    assert sonuc["answer"] == "Bilmiyorum."


def test_zincir_akis_halinde_cevap_ve_kaynak_verir():
    retriever = KaydedenRetriever(
        sorgular=[], dokumanlar=[Document(page_content="x", metadata={"source": "a.pdf", "page": 1})]
    )
    zincir = zinciri_kur(FakeListChatModel(responses=["Merhaba dünya"]), retriever)

    cevap, context = "", None
    for parca in zincir.stream({"input": "selam", "chat_history": []}):
        if "context" in parca:
            context = parca["context"]
        cevap += parca.get("answer", "")

    assert cevap == "Merhaba dünya"
    assert kaynaklari_listele(context) == ["a.pdf, sayfa 2"]


def test_zincir_gercek_chroma_ile_calisir(bellek_db):
    bellek_db.add_documents([Document(page_content="Python ile RAG", metadata={"source": "/x/n.pdf", "page": 0})])
    zincir = zinciri_kur(FakeListChatModel(responses=["Tamam."]), bellek_db.as_retriever(search_kwargs={"k": 1}))
    sonuc = zincir.invoke({"input": "RAG nedir?", "chat_history": []})
    assert sonuc["answer"] == "Tamam."
    assert kaynaklari_listele(sonuc["context"]) == ["n.pdf, sayfa 1"]


# --- veritabani_bos_mu ------------------------------------------------------

def test_veritabani_bos_mu(bellek_db):
    assert veritabani_bos_mu(bellek_db) is True
    bellek_db.add_documents([Document(page_content="a", metadata={"source": "a.pdf"})])
    assert veritabani_bos_mu(bellek_db) is False
