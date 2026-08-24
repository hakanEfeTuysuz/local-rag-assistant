from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def asistanla_konus():
    print("1. Aşama: Veritabanı ve Vektör Modeli yükleniyor...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Diske kaydettiğimiz veritabanını okuyoruz
    vektor_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    print("2. Aşama: Llama 3 Modeli hazırlanıyor...")
    # Yerelde çalışan modelimizi çağırıyoruz
    llm = Ollama(model="llama3")

    print("3. Aşama: RAG Zinciri (Chain) oluşturuluyor...\n")
    # Asıl büyü burada gerçekleşiyor: Soruyu alıp, veritabanından ilgili kısımları
    # bulup, Llama 3'e "Bu bilgilere bakarak cevap ver" diyen zinciri kuruyoruz.
    qa_zinciri = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vektor_db.as_retriever(search_kwargs={"k": 2}) # CV'den en alakalı 2 parçayı getir
    )

    # Kendi CV'mize soracağımız soruyu belirliyoruz
    soru = "CV'sinde yazan bilgilere göre Hakan Efe'nin bildiği programlama dilleri ve teknolojiler nelerdir? Lütfen Türkçe cevap ver."
    
    print(f"Soru: {soru}")
    print("Cevap düşünülüyor... (Llama 3 yerel donanımında çalışıyor, biraz sürebilir)\n")

    # Modeli çalıştırıyoruz
    cevap = qa_zinciri.invoke(soru)
    
    print("🤖 ASİSTANIN CEVABI:")
    print("-" * 50)
    print(cevap['result'])
    print("-" * 50)

if __name__ == "__main__":
    asistanla_konus()