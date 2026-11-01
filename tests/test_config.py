import importlib

import pytest

import config


@pytest.fixture
def config_yenile(monkeypatch):
    """Ortam değişkenlerini ayarlayıp config modülünü yeniden yükler; test sonunda eski haline döner."""
    def _yenile(**degiskenler):
        for isim, deger in degiskenler.items():
            monkeypatch.setenv(isim, deger)
        return importlib.reload(config)
    yield _yenile
    monkeypatch.undo()
    importlib.reload(config)


def test_varsayilanlar(config_yenile):
    c = config_yenile()
    assert c.CHUNK_SIZE == 1000
    assert c.CHUNK_OVERLAP == 200
    assert c.RETRIEVER_K >= 1


def test_ortam_degiskeni_ayari_ezer(config_yenile):
    c = config_yenile(RAG_LLM_MODEL="mistral", RAG_RETRIEVER_K="7")
    assert c.LLM_MODEL == "mistral"
    assert c.RETRIEVER_K == 7


def test_goreli_dizin_proje_klasorune_baglanir(config_yenile):
    import os
    c = config_yenile(RAG_DB_DIR="baska_db")
    assert c.DB_DIR == os.path.join(c.PROJE_DIZINI, "baska_db")


def test_gecersiz_sayi_anlasilir_hata_verir(config_yenile):
    with pytest.raises(ValueError, match="RAG_RETRIEVER_K"):
        config_yenile(RAG_RETRIEVER_K="abc")


def test_overlap_chunk_boyutundan_buyuk_olamaz(config_yenile):
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        config_yenile(RAG_CHUNK_SIZE="100", RAG_CHUNK_OVERLAP="100")
