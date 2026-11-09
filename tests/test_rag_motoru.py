import os
import shutil

import pytest

from rag_motoru import (
    dosya_hash_hesapla,
    dosya_zaten_islendi_mi,
    guvenli_dosya_adi,
    main,
    parca_kimlikleri,
    pdf_hazirla,
    vektor_veritabanina_kaydet,
)
from rag_zinciri import PdfOkumaHatasi, VeritabaniHatasi


class SahteVektorDB:
    """
    dosya_zaten_islendi_mi sadece vektor_db.get(...) çağırıp dönen sözlükteki
    'ids' listesine bakıyor. Gerçek bir Chroma kurmaya gerek yok; ayrıca
    gönderilen 'where' filtresini de kaydediyoruz ki doğru kurulduğunu görelim.
    """
    def __init__(self, kayitli_id_listesi):
        self._kayitli_id_listesi = kayitli_id_listesi
        self.son_where = None

    def get(self, where=None, **kwargs):
        self.son_where = where
        return {"ids": list(self._kayitli_id_listesi)}


# --- dosya_zaten_islendi_mi (sahte DB ile) ---------------------------------

def test_dosya_veritabaninda_varsa_true_doner():
    sahte_db = SahteVektorDB(kayitli_id_listesi=["id1", "id2"])
    assert dosya_zaten_islendi_mi(sahte_db, "cv.pdf") is True


def test_dosya_veritabaninda_yoksa_false_doner():
    sahte_db = SahteVektorDB(kayitli_id_listesi=[])
    assert dosya_zaten_islendi_mi(sahte_db, "cv.pdf") is False


def test_goreli_ve_mutlak_yol_birlikte_aranir():
    sahte_db = SahteVektorDB(kayitli_id_listesi=[])
    dosya_zaten_islendi_mi(sahte_db, "./cv.pdf")
    adaylar = sahte_db.son_where["source"]["$in"]
    assert "./cv.pdf" in adaylar
    assert "cv.pdf" in adaylar
    assert os.path.abspath("cv.pdf") in adaylar


def test_var_olan_dosyada_hash_de_aranir(ornek_pdf):
    sahte_db = SahteVektorDB(kayitli_id_listesi=[])
    dosya_zaten_islendi_mi(sahte_db, ornek_pdf)
    assert {"dosya_hash": dosya_hash_hesapla(ornek_pdf)} in sahte_db.son_where["$or"]


def test_veritabani_hatasi_anlasilir_sekilde_sarilir():
    class BozukDB:
        def get(self, **kwargs):
            raise RuntimeError("bağlantı koptu")
    with pytest.raises(VeritabaniHatasi):
        dosya_zaten_islendi_mi(BozukDB(), "cv.pdf")


# --- pdf_hazirla ------------------------------------------------------------

def test_pdf_hazirla_parcalari_ve_metadata(ornek_pdf):
    parcalar = pdf_hazirla(ornek_pdf)
    assert len(parcalar) > 0
    for p in parcalar:
        assert p.page_content.strip()
        assert p.metadata["source"] == os.path.abspath(ornek_pdf)
        assert p.metadata["page"] == 0  # ornek.pdf tek sayfalık
        assert p.metadata["dosya_hash"] == dosya_hash_hesapla(ornek_pdf)


def test_pdf_hazirla_chunk_boyutuna_uyar(ornek_pdf):
    parcalar = pdf_hazirla(ornek_pdf, chunk_size=300, chunk_overlap=50)
    assert len(parcalar) > 1
    assert all(len(p.page_content) <= 300 for p in parcalar)


def test_olmayan_dosya_hata_verir(tmp_path):
    with pytest.raises(FileNotFoundError):
        pdf_hazirla(str(tmp_path / "yok.pdf"))


def test_bozuk_pdf_anlasilir_hata_verir(tmp_path):
    bozuk = tmp_path / "bozuk.pdf"
    bozuk.write_bytes(b"bu bir pdf degil")
    with pytest.raises(PdfOkumaHatasi):
        pdf_hazirla(str(bozuk))


def test_metinsiz_pdf_anlasilir_hata_verir(tmp_path):
    from pypdf import PdfWriter
    bos = tmp_path / "bos.pdf"
    yazici = PdfWriter()
    yazici.add_blank_page(width=200, height=200)
    with open(bos, "wb") as f:
        yazici.write(f)
    with pytest.raises(PdfOkumaHatasi, match="metin çıkarılamadı"):
        pdf_hazirla(str(bos))


def _sifreli_kopya(kaynak, hedef, parola):
    from pypdf import PdfReader, PdfWriter
    yazici = PdfWriter(clone_from=PdfReader(kaynak))
    yazici.encrypt(user_password=parola, owner_password="sahip")
    with open(hedef, "wb") as f:
        yazici.write(f)
    return str(hedef)


def test_parolali_pdf_anlasilir_mesajla_reddedilir(tmp_path, ornek_pdf):
    yol = _sifreli_kopya(ornek_pdf, tmp_path / "gizli.pdf", "parola123")
    with pytest.raises(PdfOkumaHatasi, match="parola korumalı"):
        pdf_hazirla(yol)


def test_bos_parolayla_sifrelenmis_pdf_okunabilir(tmp_path, ornek_pdf):
    # Sadece kopyalama/yazdırma kısıtlı PDF'ler: açılış parolası yok, okunabilmeli
    yol = _sifreli_kopya(ornek_pdf, tmp_path / "kisitli.pdf", "")
    assert len(pdf_hazirla(yol)) == len(pdf_hazirla(ornek_pdf))


# --- Gerçek (bellekteki) Chroma ile uçtan uca tekrar-ekleme kontrolü -------

def test_ayni_dosya_farkli_yoldan_tekrar_eklenmez(bellek_db, ornek_pdf, tmp_path):
    assert dosya_zaten_islendi_mi(bellek_db, ornek_pdf) is False
    vektor_veritabanina_kaydet(pdf_hazirla(ornek_pdf), bellek_db)

    assert dosya_zaten_islendi_mi(bellek_db, ornek_pdf) is True
    # Aynı içerik, farklı klasör ve farklı isim -> yine de "işlenmiş" sayılmalı
    kopya = tmp_path / "kopya_cv.pdf"
    shutil.copy(ornek_pdf, kopya)
    assert dosya_zaten_islendi_mi(bellek_db, str(kopya)) is True


def test_eski_surumun_goreli_yolu_da_bulunur(bellek_db, sahte_embeddings):
    # Eski sürüm 'source' alanına yolu kullanıcının yazdığı gibi (göreli) kaydediyordu
    from langchain_core.documents import Document
    bellek_db.add_documents([Document(page_content="eski kayıt", metadata={"source": "cv.pdf", "page": 0})])
    assert dosya_zaten_islendi_mi(bellek_db, "./cv.pdf") is True


# --- vektor_veritabanina_kaydet: partiler, ilerleme, geri alma ------------

class PartiDB:
    """add_documents çağrılarını kaydeden; istenirse N. çağrıda hata veren sahte DB."""
    def __init__(self, hata_verecek_cagri=None):
        self.hata_verecek_cagri = hata_verecek_cagri
        self.partiler = []
        self.silinen = None

    def add_documents(self, documents, ids):
        if len(self.partiler) + 1 == self.hata_verecek_cagri:
            raise ConnectionError("Ollama koptu")
        self.partiler.append(list(ids))

    def delete(self, ids):
        self.silinen = list(ids)


def _parcalar(adet, dosya_hash="abc"):
    from langchain_core.documents import Document
    return [Document(page_content=f"p{i}", metadata={"source": "x.pdf", "page": 0, "dosya_hash": dosya_hash}) for i in range(adet)]


def test_parcalar_partiler_halinde_eklenir_ve_ilerleme_bildirilir():
    db, ilerlemeler = PartiDB(), []
    vektor_veritabanina_kaydet(_parcalar(5), db, ilerleme=lambda a, b: ilerlemeler.append((a, b)), parti_boyutu=2)
    assert [len(p) for p in db.partiler] == [2, 2, 1]
    assert ilerlemeler == [(2, 5), (4, 5), (5, 5)]


def test_hata_olursa_eklenen_parcalar_geri_silinir():
    db = PartiDB(hata_verecek_cagri=2)
    with pytest.raises(VeritabaniHatasi):
        vektor_veritabanina_kaydet(_parcalar(5), db, parti_boyutu=2)
    assert db.silinen == parca_kimlikleri(_parcalar(5))  # yarım dosya veritabanında kalmaz


def test_parca_kimlikleri_deterministiktir():
    assert parca_kimlikleri(_parcalar(3)) == ["abc-0", "abc-1", "abc-2"]
    from langchain_core.documents import Document
    rastgele = parca_kimlikleri([Document(page_content="a"), Document(page_content="b")])
    assert len(set(rastgele)) == 2


def test_ayni_dosya_iki_kez_eklense_de_kopya_olusmaz(bellek_db, ornek_pdf):
    parcalar = pdf_hazirla(ornek_pdf)
    vektor_veritabanina_kaydet(parcalar, bellek_db)
    vektor_veritabanina_kaydet(parcalar, bellek_db)
    assert len(bellek_db.get(include=[])["ids"]) == len(parcalar)


def test_yarida_kalan_ekleme_sonrasi_dosya_islenmemis_sayilir(bellek_db, ornek_pdf, monkeypatch):
    parcalar = pdf_hazirla(ornek_pdf, chunk_size=300, chunk_overlap=50)
    asil_ekle = bellek_db.add_documents
    cagri = {"n": 0}

    def ikinci_partide_bozul(docs, **kw):
        cagri["n"] += 1
        if cagri["n"] == 2:
            raise ConnectionError("Ollama koptu")
        return asil_ekle(docs, **kw)

    monkeypatch.setattr(bellek_db, "add_documents", ikinci_partide_bozul)
    with pytest.raises(VeritabaniHatasi):
        vektor_veritabanina_kaydet(parcalar, bellek_db, parti_boyutu=2)
    assert dosya_zaten_islendi_mi(bellek_db, ornek_pdf) is False  # tekrar denenebilir


# --- guvenli_dosya_adi ------------------------------------------------------

@pytest.mark.parametrize("girdi, beklenen", [
    ("cv.pdf", "cv.pdf"),
    ("../../etc/passwd.pdf", "passwd.pdf"),
    ("..\\..\\Windows\\rapor.pdf", "rapor.pdf"),
    ("/mutlak/yol/sunum.pdf", "sunum.pdf"),
    ('kötü:ad?*.pdf', "kötüad.pdf"),
    ("Özgeçmiş Hakan.pdf", "Özgeçmiş Hakan.pdf"),
])
def test_guvenli_dosya_adi(girdi, beklenen):
    assert guvenli_dosya_adi(girdi) == beklenen


def test_cok_uzun_dosya_adi_uzanti_korunarak_kisaltilir():
    ad = guvenli_dosya_adi("ş" * 300 + ".pdf")  # Türkçe harf = 2 bayt
    assert ad.endswith(".pdf")
    assert len(ad.encode("utf-8")) <= 150
    assert ad.startswith("şş")


def test_cok_uzun_adli_dosya_diske_yazilabilir(tmp_path):
    ad = guvenli_dosya_adi("b" * 400 + ".pdf")
    (tmp_path / ad).write_bytes(b"x")  # eskiden 'File name too long' hatası veriyordu


@pytest.mark.parametrize("girdi", ["", "..", "../", "   ", "a/.."])
def test_guvenli_dosya_adi_gecersiz(girdi):
    with pytest.raises(ValueError):
        guvenli_dosya_adi(girdi)


# --- CLI çıkış kodu --------------------------------------------------------

def test_cli_hata_durumunda_sifirdan_farkli_kodla_cikar(monkeypatch, tmp_path, sahte_embeddings, capsys):
    import rag_motoru
    monkeypatch.setattr(rag_motoru, "DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(rag_motoru, "embedding_yukle", lambda: sahte_embeddings)
    kod = main(["--pdf", str(tmp_path / "olmayan.pdf")])
    assert kod == 1
    assert "olmayan.pdf" in capsys.readouterr().err


def test_cli_basarili_ekleme_ve_tekrar_atlama(monkeypatch, tmp_path, sahte_embeddings, ornek_pdf):
    import rag_motoru
    monkeypatch.setattr(rag_motoru, "DB_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(rag_motoru, "embedding_yukle", lambda: sahte_embeddings)
    assert main(["--pdf", ornek_pdf]) == 0
    assert main(["--pdf", ornek_pdf]) == 0  # ikinci seferde atlanır, hata değil
