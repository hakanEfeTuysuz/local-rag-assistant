# 📚 Yerel RAG Asistanı (PDF + Ollama + LangChain)

PDF dosyalarını okuyup vektör veritabanına kaydeden ve ardından bu veritabanı üzerinden **tamamen yerel** (internet gerektirmeyen) bir LLM ile soru-cevap yapabilen basit bir RAG (Retrieval Augmented Generation) sistemi.

Sistem, embedding ve dil modeli için [Ollama](https://ollama.com) kullanır; bu sayede verileriniz hiçbir zaman bilgisayarınızdan dışarı çıkmaz.

---

## 🎯 Proje Amacı / Motivasyon

Bu proje, bulut API'lerine (OpenAI vb.) bağımlı kalmadan, tamamen yerel donanım üzerinde çalışan ve veri gizliliğini merkeze alan gerçek dünya AI çözümleri geliştirmek amacıyla bir **Proof of Concept (PoC)** olarak hazırlanmıştır. "Tutorial hell"den çıkıp, RAG mimarisini sıfırdan kurma pratiğidir.

---

## 🚀 Özellikler

- PDF dosyasını otomatik olarak okuma ve sayfalara ayırma
- Metni anlamsal parçalara bölme (chunking)
- `nomic-embed-text` modeli ile embedding oluşturma
- Embedding'leri **ChromaDB** içinde kalıcı olarak saklama
- `llama3` modeli ile veritabanı üzerinden soru-cevap (RAG)
- Terminal üzerinden `argparse` ile dinamik dosya/soru girişi
- Anlaşılır Türkçe hata mesajları ve aşama aşama ilerleme çıktıları

---

## 🗂️ Proje Yapısı

```
.
├── rag_motoru.py     # PDF'i işleyip vektör veritabanını oluşturan script
├── soru_cevap.py       # Oluşturulan veritabanına soru sormayı sağlayan script
├── chroma_db/        # Oluşturulan vektör veritabanının kaydedildiği klasör (otomatik oluşur)
└── README.md
```

> Not: Script dosya adlarını kendi projenizdeki gerçek isimlerle değiştirebilirsiniz.

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
pip install langchain langchain-community chromadb pypdf
```

İsterseniz bir `requirements.txt` dosyası oluşturup şu şekilde kurabilirsiniz:

```txt
langchain
langchain-community
chromadb
pypdf
```

```bash
pip install -r requirements.txt
```

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

---

## ▶️ Kullanım

### 1. Adım: PDF'i Vektör Veritabanına Kaydetme

`rag_motoru.py` scripti, verilen PDF dosyasını okur, parçalara böler ve `./chroma_db` klasörüne kaydeder.

```bash
python rag_motoru.py --pdf cv.pdf
```

**Parametreler:**

| Parametre | Zorunlu | Açıklama |
|---|---|---|
| `--pdf` | ✅ | İşlenecek PDF dosyasının yolu |

> ⚠️ `./chroma_db` klasörü zaten mevcutsa script yeni bir veritabanı oluşturmaz. Yeni bir PDF ile baştan başlamak isterseniz önce bu klasörü silmelisiniz:
> ```bash
> rm -rf chroma_db      # Windows: rmdir /s /q chroma_db
> ```

### 2. Adım: Asistana Soru Sorma

Veritabanı oluşturulduktan sonra `soru_cevap.py` scripti ile PDF içeriği hakkında soru sorabilirsiniz.

```bash
python soru_cevap.py -s "Bu dokümanın konusu nedir?"
```

**Parametreler:**

| Parametre | Kısa Hali | Zorunlu | Açıklama |
|---|---|---|---|
| `--soru` | `-s` | ✅ | Asistana sorulacak soru (tırnak içinde) |

Örnek çıktı:

```
Soru: Bu dokümanın konusu nedir?
Cevap düşünülüyor... (Llama 3 yerel donanımında çalışıyor, biraz sürebilir)

🤖 ASİSTANIN CEVABI:
--------------------------------------------------
...cevap burada görünür...
--------------------------------------------------
```

---

## 🛠️ Nasıl Çalışır?

1. **Yükleme:** `PyPDFLoader` ile PDF sayfa sayfa okunur.
2. **Bölme:** `RecursiveCharacterTextSplitter` ile metin, 1000 karakterlik ve 200 karakter üst üste binen (overlap) parçalara bölünür.
3. **Embedding:** Her parça, Ollama'nın `nomic-embed-text` modeli ile vektöre dönüştürülür.
4. **Saklama:** Vektörler `ChromaDB` içinde `./chroma_db` klasörüne kalıcı olarak yazılır.
5. **Sorgulama:** Kullanıcının sorusu, en alakalı `k=2` parça ile birlikte `llama3` modeline "stuff" zinciri (chain) üzerinden gönderilir ve cevap üretilir.

---

## ❗ Sorun Giderme

| Hata Mesajı | Olası Sebep | Çözüm |
|---|---|---|
| `Belirtilen '...' dosyası bulunamadı!` | PDF yolu yanlış | Dosya yolunu ve adını kontrol edin |
| `Vektör veritabanı oluşturulamadı. Ollama servisi çalışıyor mu?` | Ollama arka planda çalışmıyor | `ollama serve` komutunu çalıştırın |
| `Vektör veritabanı bulunamadı!` | Önce `rag_motoru.py` çalıştırılmamış | Önce PDF'i işleyip veritabanını oluşturun |
| Cevap üretimi çok yavaş | Yerel donanım (CPU/GPU) yetersiz | Daha küçük bir model deneyin (örn. `llama3:8b` yerine daha hafif alternatifler) |

---

## 📌 Notlar

- Bu proje tamamen **yerel** çalışır; internet bağlantısı sadece ilk kurulumda Ollama modellerini indirmek için gereklidir.
- `chroma_db` klasörü, her PDF için tek seferlik oluşturulur. Birden fazla PDF eklemek isterseniz kodu, mevcut veritabanına ekleme (append) yapacak şekilde genişletebilirsiniz.
- Cevap kalitesi kullanılan LLM modeline (`llama3`) ve chunk boyutuna göre değişebilir.

---

## 📄 Lisans

Bu proje dilediğiniz gibi kullanılabilir. Bir lisans eklemek isterseniz [MIT Lisansı](https://opensource.org/licenses/MIT) önerilir.