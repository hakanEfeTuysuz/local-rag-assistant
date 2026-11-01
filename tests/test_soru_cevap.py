import pytest

import soru_cevap
from rag_zinciri import RagHatasi, VeritabaniHatasi


class SahteZincir:
    """İlk çağrıda hata veren, sonrakilerde cevap dönen sahte zincir."""
    def __init__(self, hata_verecek_cagri_sayisi=0):
        self.kalan_hata = hata_verecek_cagri_sayisi
        self.gelen_gecmisler = []

    def invoke(self, girdi):
        self.gelen_gecmisler.append(list(girdi["chat_history"]))
        if self.kalan_hata > 0:
            self.kalan_hata -= 1
            raise ConnectionError("Ollama yanıt vermiyor")
        return {"answer": "cevap", "context": []}


def test_tek_soru_sor_gecmise_ekler():
    gecmis = []
    soru_cevap.tek_soru_sor(SahteZincir(), "merhaba", gecmis)
    assert [m.content for m in gecmis] == ["merhaba", "cevap"]


def test_tek_soru_sor_hatayi_ragHatasi_olarak_sarar():
    gecmis = []
    with pytest.raises(RagHatasi):
        soru_cevap.tek_soru_sor(SahteZincir(hata_verecek_cagri_sayisi=1), "merhaba", gecmis)
    assert gecmis == []  # başarısız soru geçmişe yazılmaz


def test_sohbet_modu_hatadan_sonra_devam_eder(monkeypatch, capsys):
    girdiler = iter(["ilk soru", "ikinci soru", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(girdiler))
    zincir = SahteZincir(hata_verecek_cagri_sayisi=1)

    soru_cevap.sohbet_modu(zincir)

    assert len(zincir.gelen_gecmisler) == 2  # hata sohbeti bitirmedi
    assert "Tekrar deneyebilir" in capsys.readouterr().out


def test_sohbet_modu_gecmisi_kirpilmis_gonderir(monkeypatch):
    monkeypatch.setattr(soru_cevap, "gecmisi_kirp", lambda g: g[-2:])
    girdiler = iter(["1", "2", "3", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(girdiler))
    zincir = SahteZincir()
    soru_cevap.sohbet_modu(zincir)
    assert [len(g) for g in zincir.gelen_gecmisler] == [0, 2, 2]


def test_veritabani_yoksa_anlasilir_hata(monkeypatch, tmp_path):
    monkeypatch.setattr(soru_cevap, "DB_DIR", str(tmp_path / "yok"))
    with pytest.raises(VeritabaniHatasi, match="bulunamadı"):
        soru_cevap.zinciri_olustur()


def test_main_hata_durumunda_1_doner(monkeypatch, tmp_path):
    monkeypatch.setattr(soru_cevap, "DB_DIR", str(tmp_path / "yok"))
    assert soru_cevap.main(["-s", "deneme"]) == 1
