import argparse
import logging
import os
import sys

from langchain_core.messages import AIMessage, HumanMessage

from config import DB_DIR, RETRIEVER_K  # MERKEZİ AYARLAR
from rag_zinciri import (
    RagHatasi,
    VeritabaniHatasi,
    cli_loglamasini_ayarla,
    gecmisi_kirp,
    kaynaklari_listele,
    modelleri_yukle,
    vektor_db_ac,
    veritabani_bos_mu,
    zinciri_kur,
)

logger = logging.getLogger(__name__)

__all__ = ["zinciri_olustur", "tek_soru_sor", "sohbet_modu", "kaynaklari_listele"]


def zinciri_olustur():
    if not os.path.exists(DB_DIR):
        raise VeritabaniHatasi(f"Vektör veritabanı bulunamadı! Önce 'rag_motoru.py' çalıştırılarak {DB_DIR} oluşturulmalıdır.")

    logger.info("1. Aşama: Modeller yükleniyor ve Ollama bağlantısı kontrol ediliyor...")
    embeddings, llm = modelleri_yukle()

    logger.info("2. Aşama: Vektör veritabanı açılıyor...")
    vektor_db = vektor_db_ac(DB_DIR, embeddings)
    if veritabani_bos_mu(vektor_db):
        raise VeritabaniHatasi(f"'{DB_DIR}' veritabanı boş! Önce 'rag_motoru.py --pdf <dosya>' ile PDF ekleyin.")
    retriever = vektor_db.as_retriever(search_kwargs={"k": RETRIEVER_K})

    logger.info("3. Aşama: LCEL Mimarisi ile RAG Zinciri oluşturuluyor...\n")
    return zinciri_kur(llm, retriever)


def tek_soru_sor(rag_zinciri, soru, chat_history):
    print(f"Soru: {soru}")
    print("Cevap düşünülüyor... (LCEL mimarisi kullanılıyor)\n")

    try:
        cevap = rag_zinciri.invoke({"input": soru, "chat_history": gecmisi_kirp(chat_history)})
    except Exception as e:
        raise RagHatasi(f"Cevap üretilirken bir hata oluştu (Ollama hâlâ çalışıyor mu?): {e}") from e

    print("🤖 ASİSTANIN CEVABI:")
    print("-" * 50)
    print(cevap["answer"])
    print("-" * 50)

    kaynaklar = kaynaklari_listele(cevap.get("context", []))
    if kaynaklar:
        print("\n📚 Kaynaklar:")
        for k in kaynaklar:
            print(f"  - {k}")

    chat_history.append(HumanMessage(content=soru))
    chat_history.append(AIMessage(content=cevap["answer"]))


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
        try:
            tek_soru_sor(rag_zinciri, soru, chat_history)
        except KeyboardInterrupt:
            print("\n(Cevap iptal edildi.)")
        except RagHatasi as hata:
            # Tek bir sorudaki hata tüm sohbeti (ve geçmişi) kaybettirmesin
            print(f"\n❌ {hata}\nTekrar deneyebilir veya 'q' ile çıkabilirsiniz.")
        print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Oluşturulan RAG veritabanı üzerinde LCEL ile soru sorar. -s verilmezse interaktif sohbet moduna geçer.")
    parser.add_argument("-s", "--soru", type=str, required=False, help="Tek seferlik soru (verilmezse sohbet modu açılır)")
    args = parser.parse_args(argv)
    cli_loglamasini_ayarla()

    try:
        rag_zinciri = zinciri_olustur()
        if args.soru:
            tek_soru_sor(rag_zinciri, args.soru, [])
        else:
            sohbet_modu(rag_zinciri)
    except RagHatasi as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
