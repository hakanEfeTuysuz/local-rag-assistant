import re

import pytest
import toml

import temalar
from temalar import TEMALAR, VARSAYILAN_TEMA, kayitli_temayi_oku, tema_uygula, temayi_kaydet, vurgu_css


@pytest.fixture(autouse=True)
def temayi_geri_al():
    """Streamlit ayarları süreç genelinde olduğu için her testten sonra varsayılan temaya dönülür."""
    yield
    tema_uygula(VARSAYILAN_TEMA)


def _paletler():
    for anahtar, tema in TEMALAR.items():
        if "palet" in tema:
            yield anahtar, tema["palet"]
        else:
            yield f"{anahtar}-acik", tema["acik"]
            yield f"{anahtar}-koyu", tema["koyu"]


def _kontrast(renk1: str, renk2: str) -> float:
    """WCAG 2 kontrast oranı."""
    def parlaklik(hex_renk):
        def kanal(c):
            c /= 255
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        r, g, b = (int(hex_renk[i:i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * kanal(r) + 0.7152 * kanal(g) + 0.0722 * kanal(b)
    a, b = sorted((parlaklik(renk1), parlaklik(renk2)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def test_varsayilan_tema_tanimli_ve_en_az_alti_tema_var():
    assert VARSAYILAN_TEMA in TEMALAR
    assert len(TEMALAR) >= 6


@pytest.mark.parametrize("ad, palet", list(_paletler()))
def test_her_palet_eksiksiz_ve_gecerli(ad, palet):
    for anahtar in (*temalar.PALET_ANAHTARLARI, ):
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", palet[anahtar]), (ad, anahtar)
    for anahtar in temalar.KENAR_ANAHTARLARI:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", palet["kenar"][anahtar]), (ad, anahtar)
    assert len(palet["vurgu"]) == 3 and all(0 <= c <= 255 for c in palet["vurgu"])


def _koyu_mu(palet) -> bool:
    return _kontrast(palet["backgroundColor"], "#000000") < _kontrast(palet["backgroundColor"], "#FFFFFF")


@pytest.mark.parametrize("ad, palet", list(_paletler()))
def test_her_temada_yazilar_okunakli(ad, palet):
    # WCAG: normal metin için AAA (7:1), bağlantılar için AA (4.5:1)
    assert _kontrast(palet["textColor"], palet["backgroundColor"]) >= 7, ad
    assert _kontrast(palet["textColor"], palet["secondaryBackgroundColor"]) >= 7, ad
    assert _kontrast(palet["textColor"], palet["kenar"]["backgroundColor"]) >= 7, ad
    assert _kontrast(palet["linkColor"], palet["backgroundColor"]) >= 4.5, ad


@pytest.mark.parametrize("ad, palet", list(_paletler()))
def test_ana_renk_hem_butonda_hem_zeminde_okunakli(ad, palet):
    ana = palet["primaryColor"]
    if _koyu_mu(palet):
        # Koyu temada ana renk hem beyaz buton yazısının zemini hem de koyu zeminde
        # (seçili etiket, kaydırıcı) yazı/çizgi rengidir; ikisini birden 4.5'e çıkarmak
        # matematiksel olarak mümkün değil, bu yüzden dengeli bir alt sınır aranır.
        assert min(_kontrast("#FFFFFF", ana), _kontrast(ana, palet["backgroundColor"]),
                   _kontrast(ana, palet["secondaryBackgroundColor"])) >= 3.8, ad
    else:
        assert _kontrast("#FFFFFF", ana) >= 4.5, ad  # birincil butondaki beyaz yazı
        assert _kontrast(ana, palet["backgroundColor"]) >= 4.5, ad


def test_sistem_temasi_config_toml_ile_ayni():
    # İlk açılışta (tema uygulanmadan önce) config.toml'daki renkler görünür; ikisi tutarlı olmalı
    ayar = toml.load(".streamlit/config.toml")["theme"]
    for mod, bolum in (("acik", "light"), ("koyu", "dark")):
        palet = TEMALAR["sistem"][mod]
        for k in temalar.PALET_ANAHTARLARI:
            assert ayar[bolum][k].upper() == palet[k].upper(), (bolum, k)
        for k in temalar.KENAR_ANAHTARLARI:
            assert ayar[bolum]["sidebar"][k].upper() == palet["kenar"][k].upper(), (bolum, "sidebar", k)


def test_tema_uygulanir_ve_tekrar_uygulamak_bir_sey_degistirmez():
    from streamlit import config

    assert tema_uygula("orman") is True
    assert config.get_option("theme.base") == "dark"
    assert config.get_option("theme.primaryColor") == TEMALAR["orman"]["palet"]["primaryColor"]
    assert config.get_option("theme.light.primaryColor") is None  # sistem paletleri temizlenir
    assert tema_uygula("orman") is False  # gereksiz yeniden çalıştırma olmaz


def test_sistem_temasina_donulunce_sabit_tema_temizlenir():
    from streamlit import config

    tema_uygula("sepya")
    tema_uygula("sistem")
    assert config.get_option("theme.base") is None
    assert config.get_option("theme.primaryColor") is None
    assert config.get_option("theme.dark.primaryColor") == TEMALAR["sistem"]["koyu"]["primaryColor"]


def test_bilinmeyen_tema_varsayilana_duser():
    tema_uygula("orman")
    tema_uygula("boyle-bir-tema-yok")
    from streamlit import config
    assert config.get_option("theme.base") is None


def test_vurgu_css():
    assert "--vurgu: 52, 211, 153" in vurgu_css("orman")
    sistem = vurgu_css("sistem")
    assert "prefers-color-scheme: dark" in sistem  # sistem temasında açık/koyu ayrı vurgu


def test_kaydet_ve_oku(tmp_path):
    dosya = tmp_path / "alt" / "tema.txt"
    assert kayitli_temayi_oku(str(dosya)) == VARSAYILAN_TEMA  # dosya yok
    temayi_kaydet(str(dosya), "lavanta")
    assert kayitli_temayi_oku(str(dosya)) == "lavanta"
    temayi_kaydet(str(dosya), "gecersiz")  # geçersiz seçim kaydedilmez
    assert kayitli_temayi_oku(str(dosya)) == "lavanta"
    dosya.write_text("bozuk-deger")
    assert kayitli_temayi_oku(str(dosya)) == VARSAYILAN_TEMA
