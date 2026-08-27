import os
import argparse
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def asistanla_konus(soru):
    # HATA YAKALAMA 1: Veritabanı yoksa patlama, uyarı ver!
    if not os.path.exists("./chroma_db"):
        raise FileNotFoundError("Vektör veritabanı bulunamadı! Önce 'rag_motoru.py' çalıştırılarak veritabanı oluşturulmalıdır.")

    print("1. Aşama: Veritabanı ve Vektör Modeli yükleniyor...")
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vektor_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    except Exception as e:
        raise Exception(f"Veritabanı yüklenirken hata oluştu: {str(e)}")

    print("2. Aşama: Llama 3 Modeli hazırlanıyor...")
    try:
        llm = Ollama(model="llama3")
        
        print("3. Aşama: RAG Zinciri (Chain) oluşturuluyor...\n")
        qa_zinciri = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vektor_db.as_retriever(search_kwargs={"k": 2})
        )
    except Exception as e:
         raise Exception(f"Model yüklenirken hata. Ollama çalışıyor mu? Hata: {str(e)}")

    print(f"Soru: {soru}")
    print("Cevap düşünülüyor... (Llama 3 yerel donanımında çalışıyor, biraz sürebilir)\n")

    try:
        cevap = qa_zinciri.invoke(soru)
        print("🤖 ASİSTANIN CEVABI:")
        print("-" * 50)
        print(cevap['result'])
        print("-" * 50)
    except Exception as e:
        raise Exception(f"Cevap üretilirken donanımsal veya sistemsel bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    # ARGPARSE KULLANIMI: Terminalden -s veya --soru parametresi ile dinamik soru alma
    parser = argparse.ArgumentParser(description="Oluşturulan RAG veritabanı üzerinde LLM'e soru sorar.")
    parser.add_argument("-s", "--soru", type=str, required=True, help="Asistana sormak istediğiniz soru (Tırnak içinde yazınız)")
    args = parser.parse_args()

    try:
        asistanla_konus(args.soru)
    except Exception as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}")