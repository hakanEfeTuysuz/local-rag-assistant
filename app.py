import hashlib
import html
import os
import shutil
import time

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import LLM_MODEL, RETRIEVER_K, TEMA_DOSYASI, UPLOAD_KLASORU, WEB_DB_DIR  # WEB_DB_DIR, CLI'nin DB_DIR'ından bilerek ayrı
from rag_motoru import guvenli_dosya_adi, pdf_hazirla, vektor_veritabanina_kaydet
from rag_zinciri import (
    RagHatasi,
    alintilari_cikar,
    gecmisi_kirp,
    kaynaklari_listele,
    modelleri_yukle,
    vektor_db_ac,
    zinciri_kur,
)
from temalar import TEMALAR, kayitli_temayi_oku, tema_uygula, temayi_kaydet, vurgu_css

KULLANICI_AVATARI = ":material/person:"
ASISTAN_AVATARI = ":material/auto_awesome:"
ORNEK_SORULAR = [
    "Bu belge ne hakkında? Kısaca özetler misin?",
    "Belgedeki en önemli 5 bilgi nedir?",
    "Belgede geçen tarih ve sayıları listeler misin?",
]

st.set_page_config(
    page_title="Yerel RAG Asistanı", page_icon="📚", layout="centered", initial_sidebar_state="auto"
)

# Seçili renk teması (temalar.py). Tema değiştiyse yeni renklerin tarayıcıya
# gitmesi için sayfa bir kez yeniden çalıştırılır.
aktif_tema = kayitli_temayi_oku(TEMA_DOSYASI)
if tema_uygula(aktif_tema):
    st.rerun()

# Ana renkler temadan gelir. Buradaki özel stiller yarı saydam renkler ve temanın
# vurgu rengini (--vurgu) kullandığı için her temada okunaklı kalır.
st.markdown(f"<style>{vurgu_css(aktif_tema)}</style>", unsafe_allow_html=True)
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 820px; }
    header[data-testid="stHeader"] { background: transparent; }

    .baslik { text-align: center; margin: 0.5rem 0 1.25rem; }
    .baslik h1 { font-size: 2.1rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -0.02em; }
    .baslik p { opacity: 0.65; margin: 0.35rem 0 0.75rem; font-size: 0.95rem; }

    .rozetler { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; }
    .rozet {
        display: inline-flex; align-items: center; gap: 0.3rem;
        padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.78rem; line-height: 1.6;
        background: rgba(var(--vurgu), 0.10); border: 1px solid rgba(var(--vurgu), 0.25);
    }
    .rozet.kaynak { font-size: 0.8rem; background: rgba(127, 127, 127, 0.08); border-color: rgba(127, 127, 127, 0.22); }
    .rozetler.sol { justify-content: flex-start; margin-top: 0.35rem; }
    .alt-bilgi { font-size: 0.8rem; opacity: 0.55; margin-left: 0.25rem; align-self: center; }

    .adim { font-size: 0.92rem; line-height: 1.45; padding-bottom: 0.75rem; }
    .adim .no {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.6rem; height: 1.6rem; border-radius: 50%; margin-bottom: 0.4rem;
        font-weight: 700; font-size: 0.85rem; background: rgba(var(--vurgu), 0.15);
    }
    .adim b { display: block; margin-bottom: 0.15rem; }
    .adim span { opacity: 0.7; }

    .belge {
        display: flex; align-items: center; gap: 0.5rem; padding: 0.45rem 0.6rem; margin-bottom: 0.35rem;
        border-radius: 0.6rem; border: 1px solid rgba(127, 127, 127, 0.18); font-size: 0.85rem;
    }
    .belge .ad { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .belge .sayi { opacity: 0.55; font-size: 0.75rem; white-space: nowrap; }
    .bolum-basligi { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.55; margin: 0.5rem 0 0.4rem; }

    .alinti { padding: 0.55rem 0.8rem; margin-bottom: 0.6rem; border-left: 3px solid rgba(var(--vurgu), 0.55);
              background: rgba(127, 127, 127, 0.06); border-radius: 0 0.5rem 0.5rem 0; }
    .alinti .kaynak-adi { font-size: 0.75rem; font-weight: 700; opacity: 0.7; margin-bottom: 0.2rem; }
    .alinti .metin { font-size: 0.88rem; line-height: 1.5; white-space: pre-wrap; }

    .dusunuyor { display: inline-flex; gap: 0.3rem; padding: 0.5rem 0; }
    .dusunuyor i { width: 0.45rem; height: 0.45rem; border-radius: 50%; background: currentColor; opacity: 0.35;
                   animation: zipla 1.2s infinite ease-in-out; }
    .dusunuyor i:nth-child(2) { animation-delay: 0.15s; }
    .dusunuyor i:nth-child(3) { animation-delay: 0.3s; }
    @keyframes zipla { 0%, 80%, 100% { transform: translateY(0); opacity: 0.35; } 40% { transform: translateY(-4px); opacity: 0.9; } }
    @media (prefers-reduced-motion: reduce) { .dusunuyor i { animation: none; } }
</style>
""", unsafe_allow_html=True)


def _rozet(metin: str, sinif: str = "") -> str:
    return f'<span class="rozet {sinif}">{html.escape(metin)}</span>'


# ---------------------------------------------------------------------------
# Kaynaklar (modeller, veritabanı) ve önbellekler
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Modeller ve veritabanı yükleniyor...")
def kaynaklari_yukle():
    """
    Modeller ve veritabanı süreç başına BİR kez açılır ve tüm sekmeler/oturumlar
    tarafından paylaşılır. Böylece bir sekmede yapılan sıfırlama diğerlerinde
    'koleksiyon bulunamadı' hatasına yol açmaz. Hata olursa önbelleğe alınmaz,
    Ollama başlatıldıktan sonra sayfa yenilenince tekrar denenir.
    """
    embeddings, llm = modelleri_yukle()
    vektor_db = vektor_db_ac(WEB_DB_DIR, embeddings)
    return llm, vektor_db


@st.cache_resource
def _veritabani_surumu() -> dict:
    """Veritabanı her değiştiğinde (ekleme/sıfırlama) artan, tüm oturumlarca paylaşılan sayaç."""
    return {"no": 0}


def veritabani_degisti() -> None:
    _veritabani_surumu()["no"] += 1


@st.cache_data(show_spinner=False, max_entries=8)
def _dosya_listesi_oku(_vektor_db, db_dizini: str, surum: int) -> list[dict]:
    """
    Kalıcı veritabanındaki parçaların metadata'sını tarayıp hangi dosyadan kaç
    parça geldiğini sayar. Tüm kayıtları okumak büyük veritabanlarında pahalı
    olduğu için sonuç önbelleğe alınır; sadece veritabanı değiştiğinde (surum
    arttığında) yeniden okunur. Eskiden kenar çubuğundaki her tıklamada okunuyordu.
    """
    kayitlar = _vektor_db.get(include=["metadatas"])
    dosyalar = {}
    for meta in kayitlar.get("metadatas") or []:
        meta = meta or {}
        isim = os.path.basename(meta.get("source", "bilinmeyen dosya"))
        kayit = dosyalar.setdefault(isim, {"isim": isim, "parca_sayisi": 0, "hashler": set()})
        kayit["parca_sayisi"] += 1
        if meta.get("dosya_hash"):
            kayit["hashler"].add(meta["dosya_hash"])
    return list(dosyalar.values())


def dosya_listesini_cikar(vektor_db) -> list[dict]:
    return _dosya_listesi_oku(vektor_db, WEB_DB_DIR, _veritabani_surumu()["no"])


@st.cache_resource(show_spinner=False, max_entries=16)
def _zinciri_getir(_llm, _vektor_db, db_dizini: str, k: int):
    """RAG zinciri her etkileşimde değil, sadece k değiştiğinde yeniden kurulur."""
    return zinciri_kur(_llm, _vektor_db.as_retriever(search_kwargs={"k": k}))


# ---------------------------------------------------------------------------
# İşlemler
# ---------------------------------------------------------------------------
def pdfleri_ekle(yuklenen_dosyalar, mevcut_dosyalar):
    """
    Yüklenen dosyaları güvenli bir adla diske yazar, işler ve veritabanına ekler.
    Sonuç mesajlarını (tür, metin) listesi olarak döndürür; bunlar st.rerun()
    sonrasında da görünsün diye session_state'te saklanır.
    """
    bildirimler = []
    mevcut_isimler = {d["isim"] for d in mevcut_dosyalar}
    mevcut_hashler = set().union(*(d["hashler"] for d in mevcut_dosyalar))

    for dosya in yuklenen_dosyalar:
        try:
            ad = guvenli_dosya_adi(dosya.name)
        except ValueError as hata:
            bildirimler.append(("error", f"❌ {hata}"))
            continue

        icerik = dosya.getvalue()
        dosya_hash = hashlib.sha256(icerik).hexdigest()
        if dosya_hash in mevcut_hashler:
            bildirimler.append(("info", f"ℹ️ '{ad}' zaten veritabanında, tekrar eklenmedi."))
            continue
        if ad in mevcut_isimler:
            bildirimler.append(("warning", f"⚠️ '{ad}' adında, içeriği farklı bir dosya zaten ekli. "
                                           "Dosyayı yeniden adlandırıp tekrar deneyin."))
            continue

        os.makedirs(UPLOAD_KLASORU, exist_ok=True)
        dosya_yolu = os.path.join(UPLOAD_KLASORU, ad)
        try:
            with open(dosya_yolu, "wb") as f:
                f.write(icerik)
            with st.spinner(f"'{ad}' okunuyor..."):
                parcalar = pdf_hazirla(dosya_yolu)
            cubuk = st.progress(0.0, text=f"'{ad}' veritabanına ekleniyor...")
            try:
                vektor_veritabanina_kaydet(
                    parcalar, vektor_db,
                    ilerleme=lambda biten, toplam, ad=ad, cubuk=cubuk: cubuk.progress(biten / toplam, text=f"'{ad}': {biten}/{toplam} parça"),
                )
            finally:
                cubuk.empty()
                veritabani_degisti()  # başarısız eklemede geri silme de veritabanını değiştirir
            mevcut_isimler.add(ad)
            mevcut_hashler.add(dosya_hash)
            bildirimler.append(("success", f"✅ '{ad}' eklendi ({len(parcalar)} parça)."))
        except (RagHatasi, OSError) as hata:
            if os.path.exists(dosya_yolu):
                os.remove(dosya_yolu)  # yarım kalan dosya diskte artık bırakmasın
            # OSError mesajı sunucudaki tam yolu içerir; kullanıcıya sadece sebebi gösterilir
            sebep = f"dosya diske yazılamadı ({hata.strerror})" if isinstance(hata, OSError) and hata.strerror else hata
            bildirimler.append(("error", f"❌ '{ad}' eklenemedi: {sebep}"))

    return bildirimler


def veritabanini_sifirla():
    """
    Klasörü silmek yerine koleksiyonu Chroma'nın kendi API'siyle sıfırlar:
    açık bir Chroma istemcisinin altından klasörü silmek, sonraki yazmalarda
    'readonly database' gibi hatalara yol açabiliyordu.
    """
    try:
        vektor_db.reset_collection()
    except Exception as hata:
        return [("error", f"❌ Veritabanı sıfırlanamadı: {hata}")]
    finally:
        veritabani_degisti()
    shutil.rmtree(UPLOAD_KLASORU, ignore_errors=True)
    st.session_state.mesajlar = []
    st.session_state.chat_history = []
    return [("success", "🗑️ Veritabanı sıfırlandı.")]


def _tema_degisti() -> None:
    """Tema seçilince kaydedilir; sayfanın başındaki tema_uygula() yeni temayı devreye alır."""
    temayi_kaydet(TEMA_DOSYASI, st.session_state.tema_secimi)


# ---------------------------------------------------------------------------
# Görünüm parçaları
# ---------------------------------------------------------------------------
def basligi_ciz(dosyalar: list[dict]) -> None:
    rozetler = [_rozet("🔒 %100 yerel"), _rozet(f"🧠 {LLM_MODEL}")]
    if dosyalar:
        toplam_parca = sum(d["parca_sayisi"] for d in dosyalar)
        rozetler.append(_rozet(f"📄 {len(dosyalar)} belge · {toplam_parca} parça"))
    st.markdown(
        '<div class="baslik"><h1>📚 Yerel RAG Asistanı</h1>'
        "<p>PDF'lerinizi yükleyin, ardından tamamen yerel çalışan bir yapay zeka ile sohbet edin.</p>"
        f'<div class="rozetler">{"".join(rozetler)}</div></div>',
        unsafe_allow_html=True,
    )


def karsilama_ciz() -> None:
    """Henüz belge yokken gösterilen 3 adımlı başlangıç rehberi."""
    st.info("Sohbete başlamak için soldaki panelden en az bir PDF ekleyin.", icon="👈")
    adimlar = [
        ("1", "PDF seçin", "Sol panelden bir veya birden fazla PDF dosyası seçin."),
        ("2", "Veritabanına ekleyin", "Belgeler parçalara ayrılıp bilgisayarınızda indekslenir."),
        ("3", "Soru sorun", "Cevaplar, kaynak sayfa ve alıntılarıyla birlikte gelir."),
    ]
    for kolon, (no, baslik, aciklama) in zip(st.columns(len(adimlar)), adimlar, strict=True):
        with kolon, st.container(border=True):
            st.markdown(
                f'<div class="adim"><div class="no">{no}</div><b>{baslik}</b><span>{aciklama}</span></div>',
                unsafe_allow_html=True,
            )


def kaynaklari_goster(kaynaklar: list[str], alintilar: list[dict], sure: float | None = None) -> None:
    """Cevabın altına kaynak rozetlerini, cevap süresini ve açılır 'Kaynak metinleri' bölümünü çizer."""
    if kaynaklar or sure is not None:
        rozetler = "".join(_rozet(f"📄 {k}", "kaynak") for k in kaynaklar)
        sure_html = f'<span class="alt-bilgi">⏱ {sure:.1f} sn</span>' if sure is not None else ""
        st.markdown(f'<div class="rozetler sol">{rozetler}{sure_html}</div>', unsafe_allow_html=True)
    if alintilar:
        with st.expander(f"Kaynak metinleri ({len(alintilar)})", icon=":material/menu_book:"):
            st.markdown(
                "".join(
                    f'<div class="alinti"><div class="kaynak-adi">{html.escape(a["kaynak"])}</div>'
                    f'<div class="metin">{html.escape(a["metin"])}</div></div>'
                    for a in alintilar
                ),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sayfa
# ---------------------------------------------------------------------------
try:
    llm, vektor_db = kaynaklari_yukle()
except RagHatasi as hata:
    basligi_ciz([])
    st.error(f"❌ {hata}")
    st.stop()

for anahtar, varsayilan in (("mesajlar", []), ("chat_history", []), ("bildirimler", []),
                            ("yukleyici_no", 0), ("bekleyen_soru", None)):
    if anahtar not in st.session_state:
        st.session_state[anahtar] = varsayilan

try:
    dosyalar = dosya_listesini_cikar(vektor_db)
except Exception as hata:
    # Veritabanı okunamıyorsa bunu "hiç PDF yok" gibi göstermek yanıltıcı olur
    basligi_ciz([])
    st.error(f"❌ Vektör veritabanı okunamadı ('{WEB_DB_DIR}'): {hata}")
    st.stop()

with st.sidebar:
    st.markdown('<div class="bolum-basligi">Belge ekle</div>', unsafe_allow_html=True)
    yuklenen_dosyalar = st.file_uploader(
        "Bir veya birden fazla PDF seçin", type="pdf", accept_multiple_files=True,
        key=f"yukleyici_{st.session_state.yukleyici_no}", label_visibility="collapsed",
    )
    if yuklenen_dosyalar and st.button("Veritabanına Ekle", key="ekle", type="primary", icon=":material/add:", width="stretch"):
        st.session_state.bildirimler = pdfleri_ekle(yuklenen_dosyalar, dosyalar)
        st.session_state.yukleyici_no += 1  # eklenen dosyalar seçim kutusundan temizlensin
        st.rerun()

    # Bir önceki çalıştırmadan kalan sonuç mesajlarını bir kez göster
    for tur, metin in st.session_state.bildirimler:
        getattr(st, tur)(metin)
    st.session_state.bildirimler = []

    if dosyalar:
        st.markdown(f'<div class="bolum-basligi">Belgeler ({len(dosyalar)})</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                f'<div class="belge" title="{html.escape(d["isim"])}">📄<span class="ad">{html.escape(d["isim"])}</span>'
                f'<span class="sayi">{d["parca_sayisi"]} parça</span></div>'
                for d in dosyalar
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Görünüm", icon=":material/palette:"):
        st.session_state.tema_secimi = aktif_tema  # başka sekmede değiştirildiyse de doğru seçili görünsün
        st.pills(
            "Renk teması", options=list(TEMALAR), format_func=lambda k: TEMALAR[k]["ad"], key="tema_secimi",
            required=True, on_change=_tema_degisti, label_visibility="collapsed",
        )
        if TEMALAR[aktif_tema].get("aciklama"):
            st.caption(TEMALAR[aktif_tema]["aciklama"])

    with st.expander("Ayarlar", icon=":material/tune:"):
        k_degeri = st.slider(
            "Her soruda getirilecek parça sayısı (k)", min_value=1, max_value=max(12, RETRIEVER_K), value=RETRIEVER_K,
            help="Yüksek değerler daha detaylı cevap verir ama bağlam penceresini zorlayabilir.",
        )


def alt_kontrolleri_ciz() -> None:
    """
    Kenar çubuğunun en altındaki sohbet/veritabanı kontrolleri. Sayfanın SONUNDA
    çizilir ki yeni gelen ilk cevaptan hemen sonra "Sohbeti Temizle" de görünsün.
    """
    with st.sidebar:
        if st.session_state.mesajlar and st.button(
            "Sohbeti Temizle", key="sohbeti_temizle", icon=":material/chat_bubble:", width="stretch"
        ):
            st.session_state.mesajlar = []
            st.session_state.chat_history = []
            st.rerun()
        if dosyalar:
            with st.popover("Veritabanını Sıfırla", key="sifirla_menusu", icon=":material/delete:", width="stretch"):
                st.markdown("Tüm belgeler ve sohbet geçmişi **kalıcı olarak** silinecek. Emin misiniz?")
                if st.button("Evet, her şeyi sil", key="sifirla_onay", type="primary", width="stretch"):
                    st.session_state.bildirimler = veritabanini_sifirla()
                    st.rerun()


basligi_ciz(dosyalar)

if not dosyalar:
    karsilama_ciz()
    alt_kontrolleri_ciz()
    st.stop()

rag_zinciri = _zinciri_getir(llm, vektor_db, WEB_DB_DIR, k_degeri)

# Gelen soru, sayfa çizilmeden önce belirlenir (giriş kutusu her zaman en altta sabit durur).
# Sadece boşluktan oluşan girişler modele gönderilmez.
soru = (st.chat_input("Belgelerinize bir soru sorun...") or "").strip()
if not soru and st.session_state.bekleyen_soru:
    soru = st.session_state.bekleyen_soru
st.session_state.bekleyen_soru = None

for mesaj in st.session_state.mesajlar:
    avatar = KULLANICI_AVATARI if mesaj["rol"] == "human" else ASISTAN_AVATARI
    with st.chat_message(mesaj["rol"], avatar=avatar):
        st.markdown(mesaj["icerik"])
        if mesaj["rol"] != "human":
            kaynaklari_goster(mesaj.get("kaynaklar", []), mesaj.get("alintilar", []), mesaj.get("sure"))

# Sohbet henüz başlamadıysa tek tıkla sorulabilecek örnek sorular
if not st.session_state.mesajlar and not soru:
    st.markdown('<div class="bolum-basligi" style="text-align:center">Örnek sorular</div>', unsafe_allow_html=True)
    with st.container(horizontal=True, horizontal_alignment="center"):
        for i, ornek in enumerate(ORNEK_SORULAR):
            if st.button(ornek, key=f"ornek_{i}", icon=":material/lightbulb:"):
                st.session_state.bekleyen_soru = ornek
                st.rerun()

if soru:
    st.session_state.mesajlar.append({"rol": "human", "icerik": soru})
    with st.chat_message("human", avatar=KULLANICI_AVATARI):
        st.markdown(soru)

    with st.chat_message("assistant", avatar=ASISTAN_AVATARI):
        bulunan = {"context": []}
        bekleme = st.empty()
        bekleme.markdown('<div class="dusunuyor"><i></i><i></i><i></i></div>', unsafe_allow_html=True)
        baslangic = time.perf_counter()

        def cevap_akisi():
            """Cevabı parça parça (token token) ekrana akıtır; kaynakları da yakalar."""
            for parca in rag_zinciri.stream({
                "input": soru,
                "chat_history": gecmisi_kirp(st.session_state.chat_history),
            }):
                if "context" in parca:
                    bulunan["context"] = parca["context"]
                if "answer" in parca:
                    bekleme.empty()  # ilk kelime gelince "düşünüyor" göstergesi kalkar
                    yield parca["answer"]

        try:
            cevap_metni = st.write_stream(cevap_akisi())
            if not isinstance(cevap_metni, str):
                cevap_metni = "".join(str(p) for p in cevap_metni)
            sure = time.perf_counter() - baslangic

            kaynaklar = kaynaklari_listele(bulunan["context"])
            alintilar = alintilari_cikar(bulunan["context"])
            kaynaklari_goster(kaynaklar, alintilar, sure)

            st.session_state.chat_history.append(HumanMessage(content=soru))
            st.session_state.chat_history.append(AIMessage(content=cevap_metni))
            st.session_state.mesajlar.append(
                {"rol": "assistant", "icerik": cevap_metni, "kaynaklar": kaynaklar, "alintilar": alintilar, "sure": sure}
            )
        except Exception as hata:
            bekleme.empty()
            st.error(f"Cevap üretilirken hata oluştu (Ollama hâlâ çalışıyor mu?): {hata}")

alt_kontrolleri_ciz()
