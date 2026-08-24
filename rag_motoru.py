from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

def pdf_hazirla(dosya_yolu):
    print("1. Aşama: PDF yükleniyor...")
    loader = PyPDFLoader(dosya_yolu)
    dokumanlar = loader.load()
    print(f"-> Başarılı! PDF toplam {len(dokumanlar)} sayfa olarak okundu.\n")

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
    
    # İndirdiğimiz yerel embedding (vektörleştirme) modelini tanımlıyoruz
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Vektörleri oluşturup diske (chroma_db klasörüne) kaydediyoruz
    vektor_db = Chroma.from_documents(
        documents=parcalar,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("-> Başarılı! Vektör veritabanı oluşturuldu ve '.chroma_db' klasörüne kaydedildi.\n")
    return vektor_db

if __name__ == "__main__":
    pdf_dosyasi = "ornek.pdf"
    
    # Eğer daha önce veritabanı oluşturulmuşsa tekrar oluşturmamak için küçük bir kontrol:
    if not os.path.exists("./chroma_db"):
        bolunmus_metinler = pdf_hazirla(pdf_dosyasi)
        vektor_veritabani = vektor_veritabanina_kaydet(bolunmus_metinler)
    else:
        print("Veritabanı zaten mevcut. Direkt soru-cevap aşamasına geçilebilir!")