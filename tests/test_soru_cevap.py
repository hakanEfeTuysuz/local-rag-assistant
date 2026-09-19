from soru_cevap import kaynaklari_listele


class SahteDoc:
    """
    kaynaklari_listele fonksiyonu sadece .metadata attribute'una bakıyor,
    gerçek bir langchain Document nesnesi kurmaya gerek yok.
    """
    def __init__(self, source, page):
        self.metadata = {"source": source, "page": page}


def test_tek_dosya_tek_sayfa():
    dokumanlar = [SahteDoc("ornek.pdf", 0)]
    sonuc = kaynaklari_listele(dokumanlar)
    assert sonuc == ["ornek.pdf, sayfa 1"]


def test_sayfa_numarasi_birle_basliyor():
    # PyPDFLoader sayfaları 0'dan sayar, kullanıcıya 1'den başlayarak gösterilmeli
    dokumanlar = [SahteDoc("cv.pdf", 4)]
    sonuc = kaynaklari_listele(dokumanlar)
    assert sonuc == ["cv.pdf, sayfa 5"]


def test_tekrarlanan_kaynaklar_temizlenir():
    dokumanlar = [
        SahteDoc("cv.pdf", 0),
        SahteDoc("cv.pdf", 0),   # aynı dosya, aynı sayfa - tekrar
        SahteDoc("rapor.pdf", 2),
    ]
    sonuc = kaynaklari_listele(dokumanlar)
    assert sonuc == ["cv.pdf, sayfa 1", "rapor.pdf, sayfa 3"]


def test_farkli_klasordeki_ayni_isimli_dosya():
    # os.path.basename kullanıldığı için tam yol farklı olsa da sadece dosya adı gösterilir
    dokumanlar = [SahteDoc("/home/kullanici/belgeler/cv.pdf", 0)]
    sonuc = kaynaklari_listele(dokumanlar)
    assert sonuc == ["cv.pdf, sayfa 1"]


def test_bos_liste():
    sonuc = kaynaklari_listele([])
    assert sonuc == []