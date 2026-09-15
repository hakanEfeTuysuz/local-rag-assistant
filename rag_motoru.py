import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # YENİ NESİL IMPORT
from langchain_chroma import Chroma
from config import EMBED_MODEL, DB_DIR  # MERKEZİ AYARLAR


def pdf_hazirla(dosya_yolu):
    print(f"1. Aşama: '{dosya_yolu}' yükleniyor...")

    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"Belirtilen '{dosya_yolu}' dosyası bulunamadı! Lütfen yolu kontrol edin.")

    try:
        loader = PyPDFLoader(dosya_yolu)
        dokumanlar = loader.load()
        print(f"-> Başarılı! PDF toplam {len(dokumanlar)} sayfa olarak okundu.\n")
    except Exception as e:
        raise Exception(f"PDF okunurken kritik bir hata oluştu: {str(e)}") from e

    print("2. Aşama: Metin parçalara bölünüyor (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    parcalar = text_splitter.split_documents(dokumanlar)
    print(f"-> Başarılı! Metin toplam {len(parcalar)} anlamsal parçaya bölündü.\n")

    return parcalar


def dosya_zaten_islendi_mi(vektor_db, dosya_yolu):
    """
    Aynı dosyanın veritabanına daha önce eklenip eklenmediğini kontrol eder.
    Chroma, her chunk'ın metadata'sında 'source' alanını PyPDFLoader'a verilen
    yolla birebir aynı şekilde saklar; bu yüzden aynı string ile arıyoruz.
    """
    sonuc = vektor_db.get(where={"source": dosya_yolu})
    return len(sonuc.get("ids", [])) > 0


def vektor_veritabanina_kaydet(parcalar, vektor_db, embeddings):
    if vektor_db is None:
        print("3. Aşama: Yeni vektör veritabanı oluşturuluyor...")
        try:
            vektor_db = Chroma.from_documents(
                documents=parcalar,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
            print(f"-> Başarılı! Vektör veritabanı oluşturuldu ve '{DB_DIR}' klasörüne kaydedildi.\n")
        except Exception as e:
            raise Exception(f"Vektör veritabanı oluşturulamadı. Ollama servisi çalışıyor mu? Detay: {str(e)}") from e
    else:
        print("3. Aşama: Mevcut vektör veritabanına ekleniyor...")
        try:
            vektor_db.add_documents(parcalar)
            print(f"-> Başarılı! {len(parcalar)} yeni parça '{DB_DIR}' veritabanına eklendi.\n")
        except Exception as e:
            raise Exception(f"Veritabanına ekleme yapılamadı. Detay: {str(e)}") from e

    return vektor_db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bir veya birden fazla PDF dosyasını RAG sistemi için vektör veritabanına kaydeder.")
    parser.add_argument("--pdf", type=str, required=True, nargs="+",
                         help="İşlenecek PDF dosyalarının yolu (birden fazla dosya için aralarına boşluk koyun, örn: --pdf cv.pdf rapor.pdf)")
    args = parser.parse_args()

    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        vektor_db = None
        if os.path.exists(DB_DIR):
            print(f"Mevcut veritabanı bulundu: '{DB_DIR}'. Yeni dosyalar buna eklenecek.\n")
            vektor_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

        for dosya_yolu in args.pdf:
            if vektor_db is not None and dosya_zaten_islendi_mi(vektor_db, dosya_yolu):
                print(f"⏭️  '{dosya_yolu}' zaten veritabanında var, atlanıyor.\n")
                continue

            bolunmus_metinler = pdf_hazirla(dosya_yolu)
            vektor_db = vektor_veritabanina_kaydet(bolunmus_metinler, vektor_db, embeddings)

    except Exception as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}")