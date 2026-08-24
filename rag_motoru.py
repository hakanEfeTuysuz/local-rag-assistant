from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def pdf_hazirla(dosya_yolu):
    print("1. Aşama: PDF yükleniyor...")
    # PyPDFLoader ile PDF dosyasını okuyoruz
    loader = PyPDFLoader(dosya_yolu)
    dokumanlar = loader.load()
    print(f"-> Başarılı! PDF toplam {len(dokumanlar)} sayfa olarak okundu.\n")

    print("2. Aşama: Metin parçalara bölünüyor (Chunking)...")
    # Metni 1000 karakterlik parçalara bölüyoruz. 
    # Anlam bütünlüğü kaybolmasın diye her parça bir öncekiyle 200 karakter örtüşecek (overlap).
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    parcalar = text_splitter.split_documents(dokumanlar)
    print(f"-> Başarılı! Metin toplam {len(parcalar)} anlamsal parçaya bölündü.")
    
    return parcalar

# Fonksiyonu test etmek için çalıştırıyoruz
if __name__ == "__main__":
    pdf_dosyasi = "ornek.pdf" # PDF adının bu olduğundan emin ol
    bolunmus_metinler = pdf_hazirla(pdf_dosyasi)