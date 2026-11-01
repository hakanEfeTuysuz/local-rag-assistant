import os
import shutil

import pytest

from rag_motoru import (
    dosya_hash_hesapla,
    dosya_zaten_islendi_mi,
    guvenli_dosya_adi,
    main,
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
