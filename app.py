import hashlib
import os
import re
import shutil

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import RETRIEVER_K, UPLOAD_KLASORU, WEB_DB_DIR  # WEB_DB_DIR, CLI'nin DB_DIR'ından bilerek ayrı
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

st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="📚", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 780px; }
    h1 { text-align: center; }
    .alt-baslik { text-align: center; opacity: 0.65; font-size: 0.9rem; margin-top: -0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Yerel RAG Asistanı")
# Sadece bu alt başlık ortalanır (eskiden CSS tüm st.caption'ları, örn. kaynak satırlarını da ortalıyordu)
st.markdown('<p class="alt-baslik">PDF\'lerinizi yükleyin, ardından tamamen yerel çalışan bir yapay zeka ile sohbet edin.</p>',
            unsafe_allow_html=True)

_MARKDOWN_OZEL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$])")


def md_kacis(metin: str) -> str:
    """PDF'ten gelen metindeki *, #, $ gibi karakterlerin markdown/LaTeX olarak yorumlanmasını engeller."""
    return _MARKDOWN_OZEL.sub(r"\\\1", metin)


def kaynaklari_goster(kaynaklar: list[str], alintilar: list[dict]) -> None:
    """Cevabın altına kaynak satırını ve açılır 'Kaynak metinleri' bölümünü çizer."""
    if kaynaklar:
        st.caption("📚 Kaynaklar: " + " · ".join(md_kacis(k) for k in kaynaklar))
    if alintilar:
        with st.expander(f"📖 Kaynak metinleri ({len(alintilar)})"):
            for i, alinti in enumerate(alintilar):
                if i:
                    st.divider()
                st.markdown(f"**{md_kacis(alinti['kaynak'])}**")
                st.markdown(md_kacis(alinti["metin"]))


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


try:
    llm, vektor_db = kaynaklari_yukle()
except RagHatasi as hata:
    st.error(f"❌ {hata}")
    st.stop()


def dosya_listesini_cikar(vektor_db):
    """
    Kalıcı veritabanındaki parçaların metadata'sını tarayıp hangi dosyadan kaç
    parça geldiğini sayar. Liste her çalıştırmada veritabanından okunduğu için
    sayfa yenilense veya başka bir sekmede değişiklik yapılsa da günceldir.
    """
    kayitlar = vektor_db.get(include=["metadatas"])

    dosyalar = {}
    for meta in kayitlar.get("metadatas") or []:
        meta = meta or {}
        isim = os.path.basename(meta.get("source", "bilinmeyen dosya"))
        kayit = dosyalar.setdefault(isim, {"isim": isim, "parca_sayisi": 0, "hashler": set()})
        kayit["parca_sayisi"] += 1
        if meta.get("dosya_hash"):
            kayit["hashler"].add(meta["dosya_hash"])
    return list(dosyalar.values())


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
            mevcut_isimler.add(ad)
            mevcut_hashler.add(dosya_hash)
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
    shutil.rmtree(UPLOAD_KLASORU, ignore_errors=True)
    st.session_state.mesajlar = []
    st.session_state.chat_history = []
    return [("success", "🗑️ Veritabanı sıfırlandı.")]


if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bildirimler" not in st.session_state:
    st.session_state.bildirimler = []

try:
    dosyalar = dosya_listesini_cikar(vektor_db)
except Exception as hata:
    # Veritabanı okunamıyorsa bunu "hiç PDF yok" gibi göstermek yanıltıcı olur
    st.error(f"❌ Vektör veritabanı okunamadı ('{WEB_DB_DIR}'): {hata}")
    st.stop()

with st.sidebar:
    st.header("📄 PDF Ekle")

    k_degeri = st.slider(
        "Her soruda getirilecek parça sayısı (k)", min_value=1, max_value=max(12, RETRIEVER_K), value=RETRIEVER_K,
        help="Yüksek değerler daha detaylı cevap verir ama bağlam penceresini zorlayabilir."
    )

    yuklenen_dosyalar = st.file_uploader(
        "Bir veya birden fazla PDF seçin", type="pdf", accept_multiple_files=True
    )

    if yuklenen_dosyalar and st.button("➕ Veritabanına Ekle", use_container_width=True):
        st.session_state.bildirimler = pdfleri_ekle(yuklenen_dosyalar, dosyalar)
        st.rerun()

    # Bir önceki çalıştırmadan kalan sonuç mesajlarını bir kez göster
    for tur, metin in st.session_state.bildirimler:
        getattr(st, tur)(metin)
    st.session_state.bildirimler = []

    if dosyalar:
        st.divider()
        st.caption("Eklenen dosyalar:")
        for d in dosyalar:
            st.success(f"✅ {d['isim']} ({d['parca_sayisi']} parça)")

        st.divider()
        if st.button("🗑️ Veritabanını Sıfırla", use_container_width=True):
            st.session_state.bildirimler = veritabanini_sifirla()
            st.rerun()

if not dosyalar:
    st.info("👈 Sohbete başlamak için soldaki panelden en az bir PDF ekleyin.")
    st.stop()

retriever = vektor_db.as_retriever(search_kwargs={"k": k_degeri})
rag_zinciri = zinciri_kur(llm, retriever)

for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["icerik"])
        kaynaklari_goster(mesaj.get("kaynaklar", []), mesaj.get("alintilar", []))

# Sadece boşluktan oluşan girişler modele gönderilmez
soru = (st.chat_input("Sorunuzu yazın...") or "").strip()

if soru:
    st.session_state.mesajlar.append({"rol": "human", "icerik": soru})
    with st.chat_message("human"):
        st.markdown(soru)

    with st.chat_message("assistant"):
        bulunan = {"context": []}

        def cevap_akisi():
            """Cevabı parça parça (token token) ekrana akıtır; kaynakları da yakalar."""
            for parca in rag_zinciri.stream({
                "input": soru,
                "chat_history": gecmisi_kirp(st.session_state.chat_history),
            }):
                if "context" in parca:
                    bulunan["context"] = parca["context"]
                if "answer" in parca:
                    yield parca["answer"]

        try:
            with st.spinner("Cevap düşünülüyor..."):
                cevap_metni = st.write_stream(cevap_akisi())
            if not isinstance(cevap_metni, str):
                cevap_metni = "".join(str(p) for p in cevap_metni)

            kaynaklar = kaynaklari_listele(bulunan["context"])
            alintilar = alintilari_cikar(bulunan["context"])
            kaynaklari_goster(kaynaklar, alintilar)

            st.session_state.chat_history.append(HumanMessage(content=soru))
            st.session_state.chat_history.append(AIMessage(content=cevap_metni))
            st.session_state.mesajlar.append(
                {"rol": "assistant", "icerik": cevap_metni, "kaynaklar": kaynaklar, "alintilar": alintilar}
            )
        except Exception as hata:
            st.error(f"Cevap üretilirken hata oluştu (Ollama hâlâ çalışıyor mu?): {hata}")
