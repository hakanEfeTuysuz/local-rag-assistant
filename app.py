import os
import streamlit as st

from rag_motoru import pdf_hazirla
from soru_cevap import kaynaklari_listele
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import EMBED_MODEL, LLM_MODEL, NUM_CTX

UPLOAD_KLASORU = "yuklenen_pdfler"
WEB_DB_DIR = "./web_chroma_db"  # CLI'nin ./chroma_db'sinden bilerek ayrı tutuluyor

st.set_page_config(page_title="Yerel RAG Asistanı", page_icon="📚", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; max-width: 780px; }
    h1 { text-align: center; }
    [data-testid="stCaptionContainer"] { text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("📚 Yerel RAG Asistanı")
st.caption("PDF'lerinizi yükleyin, ardından tamamen yerel çalışan bir yapay zeka ile sohbet edin.")


@st.cache_resource
def embeddings_ve_llm_yukle():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    llm = ChatOllama(model=LLM_MODEL, num_ctx=NUM_CTX)
    return embeddings, llm


try:
    embeddings, llm = embeddings_ve_llm_yukle()
except Exception as hata:
    st.error(f"Ollama'ya bağlanılamadı: {hata}")
    st.stop()


def dosya_listesini_cikar(vektor_db):
    """
    Kalıcı veritabanındaki tüm parçaları tarayıp, hangi dosyadan kaç parça
    geldiğini sayar. Sayfa yenilendiğinde (session_state sıfırlandığında)
    sidebar'daki 'eklenen dosyalar' listesini buradan yeniden kurarız.
    """
    try:
        kayitlar = vektor_db.get()
    except Exception:
        return []

    sayaç = {}
    for meta in kayitlar.get("metadatas", []):
        kaynak = os.path.basename(meta.get("source", "bilinmeyen dosya"))
        sayaç[kaynak] = sayaç.get(kaynak, 0) + 1

    return [{"isim": isim, "parca_sayisi": sayi} for isim, sayi in sayaç.items()]


if "vektor_db" not in st.session_state:
    st.session_state.vektor_db = Chroma(persist_directory=WEB_DB_DIR, embedding_function=embeddings)
if "dosyalar" not in st.session_state:
    st.session_state.dosyalar = dosya_listesini_cikar(st.session_state.vektor_db)
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def zinciri_kur(retriever):
    sistem_promptu = (
        "Sen, kullanıcının yüklediği belgeler üzerinden bilgi çıkaran, yerel donanımda çalışan zeki bir yapay zeka asistanısın. "
        "Sana verilen bağlam (context) İngilizce olsa bile, "
        "SEN HER ZAMAN SADECE TÜRKÇE cevap vereceksin. "
        "Cevabında tek bir İngilizce kelime veya cümle bile kullanma; "
        "bağlamdaki bilgileri kendi cümlelerinle Türkçeye çevirerek anlat. "
        "Sana verilen bağlam (context) bilgilerini kullanarak kullanıcının sorusunu detaylı ve somut şekilde cevapla; "
        "yüzeysel/genel geçmeden, bağlamdaki spesifik bilgileri (tanımlar, örnekler, sayılar) kullan. "
        "ÇOK ÖNEMLİ: Sana verilen bağlam parçaları, kullanıcının sorduğu KONUYLA uyuşmuyorsa "
        "(örneğin kullanıcı bir konu sordu ama bağlamda tamamen farklı bir konu/örnek/kod var), "
        "bunu asla doğruymuş gibi sunma. Bunun yerine açıkça şunu söyle: "
        "'Verilen bağlamda bu konuya dair [bağlamda gerçekte ne olduğunu belirt] bulunuyor, "
        "ancak sorduğunuz [X] konusuna dair doğrudan bir içerik bulamadım.' "
        "Bağlamdaki içerik ile soru arasındaki konu uyumunu her cevaptan önce kontrol et. "
        "Önceki konuşma geçmişini de dikkate alarak tutarlı cevaplar ver. "
        "Eğer cevabı bağlamda hiç bulamazsan, uydurma, sadece bilmediğini söyle.\n\n"
        "Bağlam:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", sistem_promptu),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    soru_cevap_zinciri = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, soru_cevap_zinciri)


with st.sidebar:
    st.header("📄 PDF Ekle")

    k_degeri = st.slider(
        "Her soruda getirilecek parça sayısı (k)", min_value=2, max_value=12, value=6,
        help="Yüksek değerler daha detaylı cevap verir ama bağlam penceresini zorlayabilir."
    )

    yuklenen_dosyalar = st.file_uploader(
        "Bir veya birden fazla PDF seçin", type="pdf", accept_multiple_files=True
    )

    if yuklenen_dosyalar and st.button("➕ Veritabanına Ekle", use_container_width=True):
        os.makedirs(UPLOAD_KLASORU, exist_ok=True)
        mevcut_isimler = {d["isim"] for d in st.session_state.dosyalar}

        for dosya in yuklenen_dosyalar:
            if dosya.name in mevcut_isimler:
                continue

            dosya_yolu = os.path.join(UPLOAD_KLASORU, dosya.name)
            with open(dosya_yolu, "wb") as f:
                f.write(dosya.getbuffer())

            try:
                with st.spinner(f"'{dosya.name}' işleniyor..."):
                    parcalar = pdf_hazirla(dosya_yolu)
                    st.session_state.vektor_db.add_documents(parcalar)
                st.session_state.dosyalar.append({"isim": dosya.name, "parca_sayisi": len(parcalar)})
            except Exception as hata:
                st.error(f"❌ '{dosya.name}' eklenemedi: {hata}")

        st.rerun()

    if st.session_state.dosyalar:
        st.divider()
        st.caption("Eklenen dosyalar:")
        for d in st.session_state.dosyalar:
            st.success(f"✅ {d['isim']} ({d['parca_sayisi']} parça)")

        st.divider()
        if st.button("🗑️ Veritabanını Sıfırla", use_container_width=True):
            import shutil
            del st.session_state["vektor_db"]
            shutil.rmtree(WEB_DB_DIR, ignore_errors=True)
            shutil.rmtree(UPLOAD_KLASORU, ignore_errors=True)
            st.session_state.dosyalar = []
            st.session_state.mesajlar = []
            st.session_state.chat_history = []
            st.rerun()

if not st.session_state.dosyalar:
    st.info("👈 Sohbete başlamak için soldaki panelden en az bir PDF ekleyin.")
    st.stop()

retriever = st.session_state.vektor_db.as_retriever(search_kwargs={"k": k_degeri})
rag_zinciri = zinciri_kur(retriever)

for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["icerik"])

soru = st.chat_input("Sorunuzu yazın...")

if soru:
    st.session_state.mesajlar.append({"rol": "human", "icerik": soru})
    with st.chat_message("human"):
        st.markdown(soru)

    with st.chat_message("assistant"):
        with st.spinner("Cevap düşünülüyor..."):
            try:
                cevap = rag_zinciri.invoke({
                    "input": soru,
                    "chat_history": st.session_state.chat_history,
                })
                cevap_metni = cevap["answer"]
                st.markdown(cevap_metni)

                kaynaklar = kaynaklari_listele(cevap.get("context", []))
                if kaynaklar:
                    st.caption("📚 Kaynaklar: " + ", ".join(kaynaklar))

                st.session_state.chat_history.append(HumanMessage(content=soru))
                st.session_state.chat_history.append(AIMessage(content=cevap_metni))
                st.session_state.mesajlar.append({"rol": "assistant", "icerik": cevap_metni})
            except Exception as hata:
                st.error(f"Cevap üretilirken hata oluştu: {hata}")