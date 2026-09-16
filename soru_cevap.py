import os
import argparse

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import EMBED_MODEL, LLM_MODEL, DB_DIR, RETRIEVER_K  # MERKEZİ AYARLAR


def kaynaklari_listele(dokumanlar):
    """
    Retriever'ın getirdiği doküman parçalarından benzersiz (dosya, sayfa)
    kaynak listesi çıkarır. PyPDFLoader sayfaları 0'dan başlatır,
    kullanıcıya göstermeden önce 1 ekliyoruz.
    """
    kaynaklar = []
    for doc in dokumanlar:
        dosya_adi = os.path.basename(doc.metadata.get("source", "bilinmeyen dosya"))
        sayfa = doc.metadata.get("page")
        if sayfa is not None:
            kaynaklar.append(f"{dosya_adi}, sayfa {sayfa + 1}")
        else:
            kaynaklar.append(dosya_adi)
    return list(dict.fromkeys(kaynaklar))


def zinciri_olustur():
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
        llm = ChatOllama(model=LLM_MODEL)
    except Exception as e:
        raise Exception(f"Model yüklenirken hata. Ollama çalışıyor mu? Hata: {str(e)}") from e

    print("3. Aşama: LCEL Mimarisi ile RAG Zinciri oluşturuluyor...\n")
    try:
        sistem_promptu = (
            "Sen, kullanıcının yüklediği belgeler üzerinden bilgi çıkaran, yerel donanımda çalışan zeki bir yapay zeka asistanısın. "
            "Sana verilen bağlam (context) İngilizce olsa bile, "
            "SEN HER ZAMAN SADECE TÜRKÇE cevap vereceksin. "
            "Cevabında tek bir İngilizce kelime veya cümle bile kullanma; "
            "bağlamdaki bilgileri kendi cümlelerinle Türkçeye çevirerek anlat. "
            "Sana verilen bağlam (context) bilgilerini kullanarak kullanıcının sorusunu cevapla. "
            "Önceki konuşma geçmişini de dikkate alarak tutarlı cevaplar ver. "
            "Eğer cevabı bağlamda bulamazsan, uydurma, sadece bilmediğini söyle.\n\n"
            "Bağlam:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", sistem_promptu),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        soru_cevap_zinciri = create_stuff_documents_chain(llm, prompt)
        rag_zinciri = create_retrieval_chain(retriever, soru_cevap_zinciri)
        return rag_zinciri
    except Exception as e:
        raise Exception(f"Zincir oluşturulurken hata: {str(e)}") from e


def tek_soru_sor(rag_zinciri, soru, chat_history):
    print(f"Soru: {soru}")
    print("Cevap düşünülüyor... (LCEL mimarisi kullanılıyor)\n")

    try:
        cevap = rag_zinciri.invoke({"input": soru, "chat_history": chat_history})

        print("🤖 ASİSTANIN CEVABI:")
        print("-" * 50)
        print(cevap['answer'])
        print("-" * 50)

        kaynaklar = kaynaklari_listele(cevap.get("context", []))
        if kaynaklar:
            print("\n📚 Kaynaklar:")
            for k in kaynaklar:
                print(f"  - {k}")

        chat_history.append(HumanMessage(content=soru))
        chat_history.append(AIMessage(content=cevap["answer"]))
    except Exception as e:
        raise Exception(f"Cevap üretilirken donanımsal veya sistemsel bir hata oluştu: {str(e)}") from e


def sohbet_modu(rag_zinciri):
    print("💬 Sohbet modu başladı. Konuşma geçmişi hatırlanacak. Çıkmak için 'q' yazın.\n")
    chat_history = []
    while True:
        try:
            soru = input("Sen: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGörüşürüz! 👋")
            break

        if soru.lower() in ("q", "quit", "exit", "çık"):
            print("Görüşürüz! 👋")
            break
        if not soru:
            continue

        print()
        tek_soru_sor(rag_zinciri, soru, chat_history)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oluşturulan RAG veritabanı üzerinde LCEL ile soru sorar. -s verilmezse interaktif sohbet moduna geçer.")
    parser.add_argument("-s", "--soru", type=str, required=False, help="Tek seferlik soru (verilmezse sohbet modu açılır)")
    args = parser.parse_args()

    try:
        rag_zinciri = zinciri_olustur()
        if args.soru:
            tek_soru_sor(rag_zinciri, args.soru, [])
        else:
            sohbet_modu(rag_zinciri)
    except Exception as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}")