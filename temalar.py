"""
Web arayüzünün renk temaları.

Streamlit temayı her çalıştırmanın başında tarayıcıya gönderir; burada tema
ayarları çalışma anında değiştirilip sayfa yeniden çalıştırılarak butonlar,
giriş kutuları, kenar çubuğu ve sohbet kutusu dahil arayüzün TAMAMI yeni
temaya geçirilir. Seçim küçük bir metin dosyasına kaydedilir; uygulama
yeniden başlatılsa da korunur.

Not: Streamlit'in herkese açık st.set_option'ı tema ayarlarını çalışma anında
değiştirmeye izin vermediği için iç `streamlit.config` modülü kullanılır.
Gelecekteki bir sürümde bu yol çalışmazsa tema_uygula() sessizce False döner
ve uygulama .streamlit/config.toml'daki varsayılan temayla çalışmaya devam eder.
"""
import os

PALET_ANAHTARLARI = ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor", "borderColor", "linkColor")
KENAR_ANAHTARLARI = ("backgroundColor", "secondaryBackgroundColor")
VARSAYILAN_TEMA = "sistem"

# Her palet: Streamlit renkleri + kenar çubuğu renkleri + özel öğeler (rozet,
# alıntı kenarlığı vb.) için "vurgu" rengi (R, G, B).
_SISTEM_ACIK = {
    "primaryColor": "#4F46E5", "backgroundColor": "#FFFFFF", "secondaryBackgroundColor": "#F1F5F9",
    "textColor": "#0F172A", "borderColor": "#E2E8F0", "linkColor": "#4338CA",
    "kenar": {"backgroundColor": "#F8FAFC", "secondaryBackgroundColor": "#EEF2F7"},
    "vurgu": (99, 102, 241),
}
_SISTEM_KOYU = {
    "primaryColor": "#6E71F1", "backgroundColor": "#0B1120", "secondaryBackgroundColor": "#1A2333",
    "textColor": "#E2E8F0", "borderColor": "#2A3548", "linkColor": "#A5B4FC",
    "kenar": {"backgroundColor": "#0F172A", "secondaryBackgroundColor": "#1A2333"},
    "vurgu": (129, 140, 248),
}

TEMALAR = {
    "sistem": {"ad": "🖥️ Sistem", "aciklama": "Bilgisayarınızın açık/koyu ayarını izler",
               "acik": _SISTEM_ACIK, "koyu": _SISTEM_KOYU},
    "gece": {"ad": "🌙 Gece Mavisi", "taban": "dark", "palet": {
        "primaryColor": "#6E71F1", "backgroundColor": "#0B1120", "secondaryBackgroundColor": "#18213A",
        "textColor": "#E2E8F0", "borderColor": "#26314D", "linkColor": "#A5B4FC",
        "kenar": {"backgroundColor": "#0E1529", "secondaryBackgroundColor": "#18213A"},
        "vurgu": (129, 140, 248)}},
    "okyanus": {"ad": "🌊 Okyanus", "taban": "light", "palet": {
        "primaryColor": "#0E7490", "backgroundColor": "#F4FAFB", "secondaryBackgroundColor": "#E0F2F5",
        "textColor": "#0C2D36", "borderColor": "#C9E6EC", "linkColor": "#0E7490",
        "kenar": {"backgroundColor": "#E8F5F8", "secondaryBackgroundColor": "#D5EDF2"},
        "vurgu": (8, 145, 178)}},
    "orman": {"ad": "🌲 Orman", "taban": "dark", "palet": {
        "primaryColor": "#05946B", "backgroundColor": "#0B1A14", "secondaryBackgroundColor": "#13271E",
        "textColor": "#D9F2E6", "borderColor": "#1F3A2D", "linkColor": "#6EE7B7",
        "kenar": {"backgroundColor": "#0E211A", "secondaryBackgroundColor": "#163024"},
        "vurgu": (52, 211, 153)}},
    "gunbatimi": {"ad": "🌅 Gün Batımı", "taban": "light", "palet": {
        "primaryColor": "#C2410C", "backgroundColor": "#FFF8F3", "secondaryBackgroundColor": "#FDEBDD",
        "textColor": "#3B1D0F", "borderColor": "#F6D5BF", "linkColor": "#C2410C",
        "kenar": {"backgroundColor": "#FEF1E7", "secondaryBackgroundColor": "#FBE0CC"},
        "vurgu": (234, 88, 12)}},
    "lavanta": {"ad": "💜 Lavanta", "taban": "light", "palet": {
        "primaryColor": "#7E22CE", "backgroundColor": "#FCFAFF", "secondaryBackgroundColor": "#F1E8FD",
        "textColor": "#2E1065", "borderColor": "#E4D4FA", "linkColor": "#7E22CE",
        "kenar": {"backgroundColor": "#F6F0FE", "secondaryBackgroundColor": "#EBDDFC"},
        "vurgu": (147, 51, 234)}},
    "grafit": {"ad": "🖤 Grafit", "taban": "dark", "palet": {
        "primaryColor": "#CC5E0A", "backgroundColor": "#121212", "secondaryBackgroundColor": "#1F1F1F",
        "textColor": "#E7E5E4", "borderColor": "#2E2E2E", "linkColor": "#FBBF24",
        "kenar": {"backgroundColor": "#181818", "secondaryBackgroundColor": "#242424"},
        "vurgu": (245, 158, 11)}},
    "sepya": {"ad": "📜 Sepya", "aciklama": "Uzun okumalar için göz yormayan kâğıt tonları", "taban": "light", "palet": {
        "primaryColor": "#92400E", "backgroundColor": "#F8F1E3", "secondaryBackgroundColor": "#EFE3CB",
        "textColor": "#3E2C1C", "borderColor": "#E0D0B0", "linkColor": "#7C2D12",
        "kenar": {"backgroundColor": "#F2E8D5", "secondaryBackgroundColor": "#E8D9BC"},
        "vurgu": (146, 64, 14)}},
}


def _istenen_ayarlar(anahtar: str) -> dict:
    """Seçilen temanın Streamlit config ayarlarını döndürür; kullanılmayanlar None (temizlenir)."""
    ayarlar = {"theme.base": None}
    for bolum in ("theme", "theme.light", "theme.dark"):
        ayarlar.update({f"{bolum}.{k}": None for k in PALET_ANAHTARLARI})
    for bolum in ("theme.sidebar", "theme.light.sidebar", "theme.dark.sidebar"):
        ayarlar.update({f"{bolum}.{k}": None for k in KENAR_ANAHTARLARI})

    tema = TEMALAR[anahtar]
    if "palet" in tema:
        # Tek, sabit tema: hem açık hem koyu sistem ayarında aynı görünür
        ayarlar["theme.base"] = tema["taban"]
        ayarlar.update({f"theme.{k}": tema["palet"][k] for k in PALET_ANAHTARLARI})
        ayarlar.update({f"theme.sidebar.{k}": tema["palet"]["kenar"][k] for k in KENAR_ANAHTARLARI})
    else:
        # Sistem teması: açık ve koyu paletler ayrı; tarayıcı sistem tercihine göre seçer
        for mod, bolum in (("acik", "theme.light"), ("koyu", "theme.dark")):
            ayarlar.update({f"{bolum}.{k}": tema[mod][k] for k in PALET_ANAHTARLARI})
            ayarlar.update({f"{bolum}.sidebar.{k}": tema[mod]["kenar"][k] for k in KENAR_ANAHTARLARI})
    return ayarlar


def tema_uygula(anahtar: str) -> bool:
    """
    Temayı Streamlit ayarlarına uygular. Bir şey değiştiyse True döner; çağıran
    taraf yeni temanın tarayıcıya gitmesi için st.rerun() yapmalıdır. Tema zaten
    uygulanmışsa veya Streamlit bu yola izin vermiyorsa False döner (hata vermez).
    """
    if anahtar not in TEMALAR:
        anahtar = VARSAYILAN_TEMA
    try:
        from streamlit import config

        degisecekler = {k: v for k, v in _istenen_ayarlar(anahtar).items() if config.get_option(k) != v}
        for k, v in degisecekler.items():
            config.set_option(k, v)
        return bool(degisecekler)
    except Exception:
        return False


def vurgu_css(anahtar: str) -> str:
    """Özel öğelerin (rozet, alıntı, adım numarası) kullandığı --vurgu CSS değişkenini üretir."""
    tema = TEMALAR.get(anahtar, TEMALAR[VARSAYILAN_TEMA])
    if "palet" in tema:
        return f":root {{ --vurgu: {', '.join(map(str, tema['palet']['vurgu']))}; }}"
    acik = ", ".join(map(str, tema["acik"]["vurgu"]))
    koyu = ", ".join(map(str, tema["koyu"]["vurgu"]))
    return f":root {{ --vurgu: {acik}; }} @media (prefers-color-scheme: dark) {{ :root {{ --vurgu: {koyu}; }} }}"


def kayitli_temayi_oku(dosya_yolu: str) -> str:
    """Kaydedilmiş tema seçimini okur; dosya yoksa veya geçersizse varsayılan tema döner."""
    try:
        with open(dosya_yolu, encoding="utf-8") as f:
            anahtar = f.read().strip()
    except OSError:
        return VARSAYILAN_TEMA
    return anahtar if anahtar in TEMALAR else VARSAYILAN_TEMA


def temayi_kaydet(dosya_yolu: str, anahtar: str) -> None:
    """Tema seçimini kaydeder; yazılamazsa (örn. salt okunur klasör) sessizce geçer."""
    if anahtar not in TEMALAR:
        return
    try:
        os.makedirs(os.path.dirname(dosya_yolu) or ".", exist_ok=True)
        with open(dosya_yolu, "w", encoding="utf-8") as f:
            f.write(anahtar)
    except OSError:
        pass
