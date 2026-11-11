"""
Web arayüzünün (app.py) uçtan uca testleri. Streamlit'in kendi test aracı
(AppTest) ile gerçek arayüz çalıştırılır; Ollama yerine sahte embedding ve
sahte sohbet modeli kullanıldığı için CI'da da çalışır.
"""
import os

import pytest
import streamlit as st
from langchain_core.language_models import FakeListChatModel
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture
def uygulama(monkeypatch, tmp_path, sahte_embeddings):
    """Her test için boş, izole bir web veritabanı ve sahte modellerle çalışan uygulama."""
    import config
    import rag_zinciri

    monkeypatch.setattr(config, "WEB_DB_DIR", str(tmp_path / "web_db"))
    monkeypatch.setattr(config, "UPLOAD_KLASORU", str(tmp_path / "uploads"))
    llm = FakeListChatModel(responses=["Bu CV Hakan Efe Tüysüz'e ait."])
    monkeypatch.setattr(rag_zinciri, "modelleri_yukle", lambda: (sahte_embeddings, llm))
    st.cache_resource.clear()  # önceki testin modelleri/veritabanı kullanılmasın

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    yield at
    st.cache_resource.clear()


def _pdf_ekle(at, dosyalar):
    at.sidebar.file_uploader[0].set_value(dosyalar).run()
    next(b for b in at.sidebar.button if "Ekle" in b.label).click().run()
    assert not at.exception, at.exception


def _bildirimler(at):
    return [e.value for tur in ("success", "info", "warning", "error") for e in getattr(at.sidebar, tur)]


@pytest.fixture
def pdf_bayt(ornek_pdf):
    with open(ornek_pdf, "rb") as f:
        return f.read()


def test_bos_veritabaninda_yonlendirme_gosterilir(uygulama):
    assert "en az bir PDF" in uygulama.info[0].value
    assert len(uygulama.chat_input) == 0


def _kaynak_html(cevap) -> str:
    """Cevap balonundaki kaynak rozetlerinin HTML'i."""
    return next(m.value for m in cevap.markdown if 'class="rozetler sol"' in m.value)


def test_pdf_eklenir_ve_soru_kaynak_ve_alintiyla_cevaplanir(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    assert any("cv.pdf" in m for m in _bildirimler(uygulama))

    uygulama.chat_input[0].set_value("Bu CV kimin?").run()
    assert not uygulama.exception, uygulama.exception

    cevap = uygulama.chat_message[-1]
    assert cevap.markdown[0].value == "Bu CV Hakan Efe Tüysüz'e ait."
    kaynaklar = _kaynak_html(cevap)
    assert "📄 cv.pdf, sayfa 1" in kaynaklar
    assert "⏱" in kaynaklar  # cevap süresi gösterilir
    alintilar = [m.value for m in cevap.markdown if 'class="alinti"' in m.value]
    assert alintilar and "cv.pdf, sayfa 1" in alintilar[0]  # "Kaynak metinleri" bölümü


def test_eklenen_dosyalar_yukleme_kutusundan_temizlenir_ve_listede_gorunur(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    assert not uygulama.sidebar.file_uploader[0].value  # aynı dosya yanlışlıkla tekrar eklenmesin
    assert any('class="belge"' in m.value and "cv.pdf" in m.value for m in uygulama.sidebar.markdown)
    assert any("1 belge · 4 parça" in m.value for m in uygulama.markdown)  # başlıktaki istatistik rozeti


def test_kaynaklar_ve_alintilar_sonraki_etkilesimde_kaybolmaz(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    uygulama.chat_input[0].set_value("Bu CV kimin?").run()
    uygulama.sidebar.slider[0].set_value(2).run()  # başka bir etkileşim -> yeniden çizim
    cevap = uygulama.chat_message[-1]
    assert _kaynak_html(cevap)
    assert any('class="alinti"' in m.value for m in cevap.markdown)


def test_ornek_soru_tiklaninca_sorulur_ve_ornekler_kaybolur(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    ornek = next(b for b in uygulama.button if b.key == "ornek_0")
    ornek.click().run()
    assert not uygulama.exception, uygulama.exception
    assert uygulama.chat_message[0].markdown[0].value.startswith("Bu belge ne hakkında")
    assert uygulama.chat_message[1].markdown[0].value == "Bu CV Hakan Efe Tüysüz'e ait."
    uygulama.run()
    assert not any(b.key and b.key.startswith("ornek_") for b in uygulama.button)


def test_sohbeti_temizle_gecmisi_siler_belgeleri_korur(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    uygulama.chat_input[0].set_value("Bu CV kimin?").run()
    next(b for b in uygulama.sidebar.button if b.key == "sohbeti_temizle").click().run()
    assert len(uygulama.chat_message) == 0
    assert uygulama.session_state.chat_history == []
    assert any("cv.pdf" in m.value for m in uygulama.sidebar.markdown)  # belge hâlâ ekli


def test_sadece_bosluktan_olusan_soru_gonderilmez(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    uygulama.chat_input[0].set_value("     ").run()
    assert not uygulama.exception
    assert len(uygulama.chat_message) == 0


def test_kotu_niyetli_ve_cok_uzun_dosya_adlari_guvenle_kaydedilir(uygulama, pdf_bayt, tmp_path):
    _pdf_ekle(uygulama, [
        ("../../kotu.pdf", pdf_bayt, "application/pdf"),
        ("ç" * 300 + ".pdf", pdf_bayt + b"\n%farkli", "application/pdf"),
    ])
    bildirimler = _bildirimler(uygulama)
    assert not any("eklenemedi" in b for b in bildirimler), bildirimler
    kaydedilenler = os.listdir(tmp_path / "uploads")
    assert "kotu.pdf" in kaydedilenler
    assert all(len(ad.encode("utf-8")) <= 150 for ad in kaydedilenler)
    assert not (tmp_path / "kotu.pdf").exists()


def test_ayni_icerik_tekrar_eklenmez_ve_onayli_sifirlama_calisir(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("cv.pdf", pdf_bayt, "application/pdf")])
    _pdf_ekle(uygulama, [("kopya.pdf", pdf_bayt, "application/pdf")])
    assert any("zaten veritabanında" in b for b in _bildirimler(uygulama))

    # Sıfırlama tek tıkla değil, açılan menüdeki onay butonuyla yapılır
    next(b for b in uygulama.sidebar.button if b.key == "sifirla_onay").click().run()
    assert not uygulama.exception
    assert "en az bir PDF" in uygulama.info[0].value
    assert any("sıfırlandı" in b for b in _bildirimler(uygulama))


def test_sifirlamadan_sonra_belge_listesi_onbellekten_eski_veri_gostermez(uygulama, pdf_bayt):
    # Önbellek parça sayısına göre değil sürüm sayacına göre yenilenir: sıfırlayıp
    # aynı parça sayısında farklı bir dosya eklenince eski dosya adı görünmemeli.
    _pdf_ekle(uygulama, [("eski.pdf", pdf_bayt, "application/pdf")])
    next(b for b in uygulama.sidebar.button if b.key == "sifirla_onay").click().run()
    _pdf_ekle(uygulama, [("yeni.pdf", pdf_bayt, "application/pdf")])
    belge_listesi = " ".join(m.value for m in uygulama.sidebar.markdown if 'class="belge"' in m.value)
    assert "yeni.pdf" in belge_listesi and "eski.pdf" not in belge_listesi


def test_dosya_adindaki_ozel_karakterler_html_olarak_kacislanir(uygulama, pdf_bayt):
    _pdf_ekle(uygulama, [("rapor_&_son'hali.pdf", pdf_bayt, "application/pdf")])
    uygulama.chat_input[0].set_value("Bu CV kimin?").run()
    kaynaklar = _kaynak_html(uygulama.chat_message[-1])
    assert "rapor_&amp;_son&#x27;hali.pdf" in kaynaklar
