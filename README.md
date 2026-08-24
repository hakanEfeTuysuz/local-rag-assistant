# Local RAG Assistant 🤖

Bu proje, tamamen yerel donanım üzerinde çalışan, bulut bağımsız bir RAG (Retrieval-Augmented Generation) pipeline uygulamasıdır. Herhangi bir dış API'ye veri göndermeden, PDF dosyaları üzerinden bilgi çıkarımı (inference) yapmayı sağlar.

## 🚀 Proje Amacı
Veri gizliliğinin kritik olduğu senaryolarda, yerel LLM (Büyük Dil Modeli) ve vektör veritabanı kullanarak "Tutorial Hell"den uzak, sahada kullanılabilecek gerçek bir AI çözümü sunmak.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python
* **Framework:** LangChain
* **LLM Engine:** Ollama (Llama 3 - 8B)
* **Embedding Modeli:** Nomic-Embed-Text
* **Vektör Veritabanı:** ChromaDB

## 🧠 Nasıl Çalışır?
1. **Veri Yükleme & Parçalama:** `rag_motoru.py` belgenizi (PDF) okur ve anlam bütünlüğünü koruyarak (chunking & overlap) parçalara böler.
2. **Vektörel Dönüşüm:** Metin parçaları `nomic-embed-text` ile vektörleştirilip yerel `ChromaDB` veritabanına kaydedilir.
3. **Sorgulama & Üretim:** `soru_cevap.py`, kullanıcının sorusuna en uygun metin parçalarını veritabanından çeker ve Llama 3 modeline bağlam (context) olarak sunarak anlamlı cevaplar üretir.

## ⚙️ Kurulum
Projeyi çalıştırmak için sisteminizde [Ollama](https://ollama.com/) kurulu olmalıdır.
