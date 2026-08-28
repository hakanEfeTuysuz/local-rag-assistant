import os
import argparse
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import EMBED_MODEL, LLM_MODEL, DB_DIR, RETRIEVER_K  # MERKEZİ AYARLAR

def asistanla_konus(soru):
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(f"Vektör veritabanı bulunamadı! Önce 'rag_motoru.py' çalıştırılarak {DB_DIR} oluşturulmalıdır.")

    print("1. Aşama: Veritabanı ve Vektör Modeli yükleniyor...")
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vektor_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        retriever = vektor_db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    except Exception as e:
        raise Exception(f"Veritabanı yüklenirken hata oluştu: {str(e)}") from e

    print("2. Aşama: Llama 3 Modeli hazırlanıyor...")
    try:
        # Eski nesil 'Ollama' yerine yeni nesil 'OllamaLLM' kullanıyoruz
        llm = ChatOllama(model=LLM_MODEL)
    except Exception as e:
         raise Exception(f"Model yüklenirken hata. Ollama çalışıyor mu? Hata: {str(e)}") from e

    print("3. Aşama: LCEL Mimarisi ile RAG Zinciri oluşturuluyor...\n")
    try:
        # LCEL Mimarisi: Sisteme "Genel Amaçlı ve Çok Dilli" bir asistan olduğunu öğretiyoruz
        sistem_promptu = (
        "Sen, kullanıcının yüklediği belgeler üzerinden bilgi çıkaran, yerel donanımda çalışan zeki bir yapay zeka asistanısın. "
        "Sana verilen bağlam (context) İngilizce olsa bile, "
        "SEN HER ZAMAN SADECE TÜRKÇE cevap vereceksin. "
        "Cevabında tek bir İngilizce kelime veya cümle bile kullanma; "
        "bağlamdaki bilgileri kendi cümlelerinle Türkçeye çevirerek anlat. "
        "Sana verilen bağlam (context) bilgilerini kullanarak kullanıcının sorusunu cevapla. "
        "Eğer cevabı bağlamda bulamazsan, uydurma, sadece bilmediğini söyle.\n\n"
        "Bağlam:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", sistem_promptu),
            ("human", "{input}"),
        ])

        # Yeni nesil zincirleri (Chain) birbirine bağlıyoruz
        soru_cevap_zinciri = create_stuff_documents_chain(llm, prompt)
        rag_zinciri = create_retrieval_chain(retriever, soru_cevap_zinciri)

    except Exception as e:
         raise Exception(f"Zincir oluşturulurken hata: {str(e)}") from e

    print(f"Soru: {soru}")
    print("Cevap düşünülüyor... (LCEL mimarisi kullanılıyor)\n")

    try:
        # LCEL yapısında invoke içine bir sözlük (dictionary) gönderilir
        cevap = rag_zinciri.invoke({"input": soru})
        print("🤖 ASİSTANIN CEVABI:")
        print("-" * 50)
        # Cevabın geldiği key artık 'result' değil 'answer'dır
        print(cevap['answer'])
        print("-" * 50)
    except Exception as e:
        raise Exception(f"Cevap üretilirken donanımsal veya sistemsel bir hata oluştu: {str(e)}") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oluşturulan RAG veritabanı üzerinde yeni nesil LCEL ile soru sorar.")
    parser.add_argument("-s", "--soru", type=str, required=True, help="Asistana sormak istediğiniz soru (Tırnak içinde yazınız)")
    args = parser.parse_args()

    try:
        asistanla_konus(args.soru)
    except Exception as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}")