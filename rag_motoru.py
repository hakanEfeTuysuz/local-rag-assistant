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
        # SENIOR DOKUNUŞU: 'from e' ile gerçek hata zincirini koruyoruz
        raise Exception(f"PDF okunurken kritik bir hata oluştu: {str(e)}") from e

    print("2. Aşama: Metin parçalara bölünüyor (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    parcalar = text_splitter.split_documents(dokumanlar)
    print(f"-> Başarılı! Metin toplam {len(parcalar)} anlamsal parçaya bölündü.\n")
    
    return parcalar

def vektor_veritabanina_kaydet(parcalar):
    print("3. Aşama: Metinler vektörlere çevriliyor ve veritabanına kaydediliyor...")
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vektor_db = Chroma.from_documents(
            documents=parcalar,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
 
        print(f"-> Başarılı! Vektör veritabanı oluşturuldu ve '{DB_DIR}' klasörüne kaydedildi.\n")
        return vektor_db
    except Exception as e:
         raise Exception(f"Vektör veritabanı oluşturulamadı. Ollama servisi çalışıyor mu? Detay: {str(e)}") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF dosyasını RAG sistemi için vektör veritabanına kaydeder.")
    parser.add_argument("--pdf", type=str, required=True, help="İşlenecek PDF dosyasının yolu (örn: cv.pdf)")
    args = parser.parse_args()
    
    try:
        if not os.path.exists(DB_DIR):
            bolunmus_metinler = pdf_hazirla(args.pdf)
            vektor_veritabani = vektor_veritabanina_kaydet(bolunmus_metinler)
        else:
            print(f"Uyarı: '{DB_DIR}' klasörü zaten mevcut. Yeni PDF eklemek istiyorsanız önce bu klasörü silin.")
    except Exception as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}")