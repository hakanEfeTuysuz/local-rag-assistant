# 📚 Yerel RAG Asistanı (PDF + Ollama + LangChain)

PDF dosyalarını okuyup vektör veritabanına kaydeden ve ardından bu veritabanı üzerinden **tamamen yerel** (internet gerektirmeyen) bir LLM ile soru-cevap yapabilen, LCEL mimarisiyle inşa edilmiş bir RAG (Retrieval Augmented Generation) sistemi.

Sistem, embedding ve dil modeli için [Ollama](https://ollama.com) kullanır; bu sayede verileriniz hiçbir zaman bilgisayarınızdan dışarı çıkmaz.

---

## 🎯 Proje Amacı / Motivasyon

Bu proje, bulut API'lerine (OpenAI vb.) bağımlı kalmadan, tamamen yerel donanım üzerinde çalışan ve veri gizliliğini merkeze alan gerçek dünya AI çözümleri geliştirmek amacıyla bir **Proof of Concept (PoC)** olarak hazırlanmıştır. "Tutorial hell"den çıkıp, RAG mimarisini sıfırdan kurma ve modern LangChain sürümlerine (1.x) taşıma pratiğidir.

---

## 🚀 Özellikler

- PDF dosyasını otomatik olarak okuma ve sayfalara ayırma
- Metni anlamsal parçalara bölme (chunking)
- `nomic-embed-text` modeli ile embedding oluşturma
- Embedding'leri **ChromaDB** içinde kalıcı olarak saklama (otomatik persist)
- **LCEL (LangChain Expression Language)** mimarisiyle kurulmuş modern RAG zinciri
- `ChatOllama` ile chat-formatlı prompt kullanımı (sistem/kullanıcı rolleri)
- Sistem promptu ile **her koşulda Türkçe yanıt** garantisi — bağlam İngilizce olsa bile
- Model, dizin ve retriever ayarlarının `config.py` üzerinden merkezi yönetimi
- Terminal üzerinden `argparse` ile dinamik dosya/soru girişi
- Anlaşılır Türkçe hata mesajları, `raise ... from e` ile korunmuş hata zinciri

---

## 🗂️ Proje Yapısı

```
.
├── rag_motoru.py       # PDF'i işleyip vektör veritabanını oluşturan script
├── soru_cevap.py       # Oluşturulan veritabanına LCEL zinciriyle soru sordurur
├── config.py           # Model isimleri, dizin ve retriever ayarları (tek nokta)
├── requirements.txt    # Çalışan ortamın gerçek bağımlılık sürümleri (pip freeze)
├── .gitignore           # venv/, chroma_db/, __pycache__/ hariç tutulur
├── chroma_db/           # Vektör veritabanı (otomatik oluşur, repoya dahil edilmez)
└── README.md
```

---

## ⚙️ Gereksinimler

- Python 3.9+
- [Ollama](https://ollama.com) kurulu ve çalışır durumda
- Aşağıdaki Ollama modelleri indirilmiş olmalı:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3
  ```

### Python Bağımlılıkları

```bash
pip install -r requirements.txt
```

`requirements.txt` içinde önemli olan paketler:

```txt
langchain
langchain-classic       # create_retrieval_chain, create_stuff_documents_chain
langchain-core
langchain-text-splitters
langchain-community     # PyPDFLoader
langchain-ollama        # OllamaEmbeddings, ChatOllama
langchain-chroma        # Chroma (otomatik persist eden yeni nesil paket)
chromadb
pypdf
```

> 💡 Bu dosya `pip freeze > requirements.txt` ile üretildiği için gerçekte çalışan ortamın birebir kopyasıdır — sürüm tutarsızlığı riski yoktur.

---

## 🔧 Kurulum

1. Bu depoyu klonlayın veya dosyaları bir klasöre indirin.
2. Sanal ortam oluşturmanız önerilir:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Bağımlılıkları kurun:
   ```bash
   pip install -r requirements.txt
   ```
4. Ollama servisinin arka planda çalıştığından emin olun:
   ```bash
   ollama serve
   ```
5. `config.py` içindeki model isimlerini kendi Ollama kurulumunuza göre kontrol edin:
   ```python
   EMBED_MODEL = "nomic-embed-text"
   LLM_MODEL = "llama3"
   DB_DIR = "./chroma_db"
   RETRIEVER_K = 2
   ```

---

## ▶️ Kullanım

### 1. Adım: PDF'i Vektör Veritabanına Kaydetme

`rag_motoru.py` scripti, verilen PDF dosyasını okur, parçalara böler ve `config.py`'de tanımlı dizine (`./chroma_db`) kaydeder.

```bash
python rag_motoru.py --pdf cv.pdf
```

| Parametre | Zorunlu | Açıklama |
|---|---|---|
| `--pdf` | ✅ | İşlenecek PDF dosyasının yolu |

> ⚠️ `chroma_db` klasörü zaten mevcutsa script yeni bir veritabanı oluşturmaz. Yeni bir PDF ile baştan başlamak isterseniz önce bu klasörü silin:
> ```bash
> rm -rf chroma_db      # Windows: rmdir /s /q chroma_db
> ```

### 2. Adım: Asistana Soru Sorma

Veritabanı oluşturulduktan sonra `soru_cevap.py` scripti ile PDF içeriği hakkında soru sorabilirsiniz.

```bash
python soru_cevap.py -s "Bu adayın projeleri nelerdir kısaca özetler misin?"
```

| Parametre | Kısa Hali | Zorunlu | Açıklama |
|---|---|---|---|
| `--soru` | `-s` | ✅ | Asistana sorulacak soru (tırnak içinde) |

Örnek çıktı:

```
Soru: Bu adayın projeleri nelerdir kısaca özetler misin?
Cevap düşünülüyor... (LCEL mimarisi kullanılıyor)

🤖 ASİSTANIN CEVABI:
--------------------------------------------------
Adayın projeleri şunlardır:
* IoT Tabanlı Arama Kurtarma ve Keşif Aracı: ...
--------------------------------------------------
```

---

## 🛠️ Nasıl Çalışır?

1. **Yükleme:** `PyPDFLoader` ile PDF sayfa sayfa okunur.
2. **Bölme:** `RecursiveCharacterTextSplitter` ile metin, 1000 karakterlik ve 200 karakter üst üste binen (overlap) parçalara bölünür.
3. **Embedding:** Her parça, Ollama'nın `nomic-embed-text` modeli ile vektöre dönüştürülür.
4. **Saklama:** Vektörler `langchain-chroma` paketiyle `./chroma_db` klasörüne otomatik olarak kalıcı yazılır (manuel `.persist()` çağrısına gerek yoktur; Chroma 0.4.x'ten itibaren bu otomatiktir).
5. **Zincir Kurulumu (LCEL):** `create_stuff_documents_chain` ile bir cevap-üretme zinciri, `create_retrieval_chain` ile de bu zinciri retriever'a bağlayan tam RAG zinciri oluşturulur.
6. **Sorgulama:** Kullanıcının sorusu, en alakalı `k=2` parça ile birlikte `ChatOllama` üzerinden `llama3` modeline chat formatında (`system`/`human` rolleri) gönderilir.
7. **Dil Kontrolü:** Sistem promptu, bağlam İngilizce olsa dahi modelin **daima Türkçe** yanıt vermesini zorunlu kılar.

---

## ❗ Sorun Giderme

| Hata Mesajı | Olası Sebep | Çözüm |
|---|---|---|
| `Belirtilen '...' dosyası bulunamadı!` | PDF yolu yanlış | Dosya yolunu ve adını kontrol edin |
| `Vektör veritabanı oluşturulamadı. Ollama servisi çalışıyor mu?` | Ollama arka planda çalışmıyor | `ollama serve` komutunu çalıştırın |
| `Vektör veritabanı bulunamadı!` | Önce `rag_motoru.py` çalıştırılmamış | Önce PDF'i işleyip veritabanını oluşturun |
| `ModuleNotFoundError: No module named 'langchain.chains'` | LangChain 1.x kurulu; eski chain fonksiyonları `langchain_classic`'e taşındı | `pip install langchain-classic` kurup importları `langchain_classic.chains`'ten yapın |
| `AttributeError: 'Chroma' object has no attribute 'persist'` | Yeni `langchain-chroma` paketinde `.persist()` kaldırıldı (otomatik persist var) | Kodda `.persist()` çağrısını silin |
| Model konu dışı/rol yapan cevaplar veriyor (ör. kendi "Human:" repliğini uyduruyor) | `OllamaLLM` (completion modeli) ile `ChatPromptTemplate` (chat formatı) uyumsuzluğu | `OllamaLLM` yerine `ChatOllama` kullanın |
| Türkçe soruya İngilizce cevap geliyor | Bağlam (PDF içeriği) İngilizce; model dil sinyalini bağlamdan alıyor | Sistem promptunda "daima Türkçe cevap ver" talimatını net ve vurgulu şekilde tekrarlayın; yetmezse Türkçe talimat takibi daha güçlü bir model (ör. `qwen2.5`) deneyin |
| Cevap üretimi çok yavaş | Yerel donanım (CPU/GPU) yetersiz | Daha küçük/hafif bir model deneyin |

---

## 📌 Notlar

- Bu proje tamamen **yerel** çalışır; internet bağlantısı sadece ilk kurulumda Ollama modellerini indirmek için gereklidir.
- `chroma_db` klasörü her PDF için tek seferlik oluşturulur. Birden fazla PDF eklemek isterseniz kodu, mevcut veritabanına ekleme (append) yapacak şekilde genişletebilirsiniz.
- Cevap kalitesi kullanılan LLM modeline (`llama3`), chunk boyutuna ve `RETRIEVER_K` değerine göre değişebilir.
- Proje, LangChain'in hızlı sürüm geçişlerine (0.1 → 0.3 → 1.x) uyum sağlayacak şekilde güncel tutulmuştur; `langchain_classic` gibi paket taşımalarını takip etmek gelecekte de gerekebilir.

---

## 📄 Lisans

Bu proje dilediğiniz gibi kullanılabilir. Bir lisans eklemek isterseniz [MIT Lisansı](https://opensource.org/licenses/MIT) önerilir.