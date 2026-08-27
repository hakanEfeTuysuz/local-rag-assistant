import os
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def pdf_hazirla(dosya_yolu):
    print(f"1. Aşama: '{dosya_yolu}' yükleniyor...")
    
    # HATA YAKALAMA 1: Dosya var mı kontrolü
    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"Belirtilen '{dosya_yolu}' dosyası bulunamadı! Lütfen yolu kontrol edin.")
        
    try:
        loader = PyPDFLoader(dosya_yolu)
        dokumanlar = loader.load()
        print(f"-> Başarılı! PDF toplam {len(dokumanlar)} sayfa olarak okundu.\n")
    except Exception as e:
        raise Exception(f"PDF okunurken kritik bir hata oluştu: {str(e)}")

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
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vektor_db = Chroma.from_documents(
            documents=parcalar,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("-> Başarılı! Vektör veritabanı oluşturuldu ve '.chroma_db' klasörüne kaydedildi.\n")
        return vektor_db
    # HATA YAKALAMA 2: Ollama servisi kapalıysa uyarı ver
    except Exception as e:
         raise Exception(f"Vektör veritabanı oluşturulamadı. Ollama servisi çalışıyor mu? Detay: {str(e)}")

if __name__ == "__main__":
    # ARGPARSE KULLANIMI: Dışarıdan dinamik parametre alma
    parser = argparse.ArgumentParser(description="PDF dosyasını RAG sistemi için vektör veritabanına kaydeder.")
    parser.add_argument("--pdf", type=str, required=True, help="İşlenecek PDF dosyasının yolu (örn: cv.pdf)")
    args = parser.parse_args()
    
    try:
        if not os.path.exists("./chroma_db"):
            bolunmus_metinler = pdf_hazirla(args.pdf)
            vektor_veritabani = vektor_veritabanina_kaydet(bolunmus_metinler)
        else:
            print("Uyarı: './chroma_db' klasörü zaten mevcut. Yeni PDF eklemek istiyorsanız önce bu klasörü silin.")
    except Exception as hata:
        # Hataları kırmızı çarpı ile zarifçe ekrana basıyoruz
        print(f"\n❌ SİSTEM HATASI: {hata}")