# 📚 Yerel RAG Asistanı (PDF + Ollama + LangChain)

PDF dosyalarını okuyup vektör veritabanına kaydeden ve ardından bu veritabanı üzerinden **tamamen yerel** (internet gerektirmeyen) bir LLM ile soru-cevap yapabilen, LCEL mimarisiyle inşa edilmiş bir RAG (Retrieval Augmented Generation) sistemi. Tarayıcıda açılan bir **Streamlit web arayüzü** ile PDF yükleyip anında sohbet edebilir, isterseniz terminal (CLI) üzerinden de kullanabilirsiniz.

Sistem, embedding ve dil modeli için [Ollama](https://ollama.com) kullanır; bu sayede verileriniz hiçbir zaman bilgisayarınızdan dışarı çıkmaz.

---

## 🎯 Proje Amacı / Motivasyon

Bu proje, bulut API'lerine (OpenAI vb.) bağımlı kalmadan, tamamen yerel çalışan ve veri gizliliğini merkeze alan gerçek dünya AI çözümleri geliştirmek amacıyla bir **Proof of Concept (PoC)** olarak hazırlanmıştır. "Tutorial hell"den çıkıp, RAG mimarisini sıfırdan kurma ve modern LangChain sürümlerine (1.x) taşıma pratiğidir.

---

## 🚀 Özellikler

- **Streamlit web arayüzü**: tarayıcıdan sürükle-bırak PDF yükleme, ayarlanabilir arama derinliği (k), kalıcı oturum ve tek tuşla veritabanı sıfırlama — kurulumdan sonra tek komutla açılır
- PDF dosyasını otomatik olarak okuma ve sayfalara ayırma
- Metni anlamsal parçalara bölme (chunking)
- `nomic-embed-text` modeli ile embedding oluşturma
- Embedding'leri **ChromaDB** içinde kalıcı olarak saklama (otomatik persist)
- **Çoklu PDF desteği**: veritabanı zaten varsa yeni dosyalar üstüne **eklenir** (append), aynı dosya tekrar verilirse otomatik atlanır
- **Kaynak gösterme**: her cevabın altında, bilginin hangi PDF dosyasından ve hangi sayfadan geldiği listelenir
- **Sohbet hafızası**: konuşma geçmişi hatırlanır; "onun", "bunun" gibi zamirler önceki cevaplara doğru bağlanır
- **Konu uyumu kontrolü**: getirilen bağlam, sorulan konuyla örtüşmüyorsa model bunu doğruymuş gibi sunmak yerine açıkça belirtir
- **LCEL (LangChain Expression Language)** mimarisiyle kurulmuş modern RAG zinciri
- `ChatOllama` ile chat-formatlı prompt kullanımı (sistem/kullanıcı rolleri)
- Sistem promptu ile **her koşulda Türkçe yanıt** garantisi — bağlam İngilizce olsa bile
- Model, dizin, bağlam penceresi ve retriever ayarlarının `config.py` üzerinden merkezi yönetimi
- Anlaşılır Türkçe hata mesajları, `raise ... from e` ile korunmuş hata zinciri

---

## 🗂️ Proje Yapısı

```
.
├── app.py              # Streamlit web arayüzü: PDF yükleme + sohbet, tarayıcı üzerinden
├── rag_motoru.py       # PDF'leri işleyip vektör veritabanını oluşturan/güncelleyen script (CLI)
├── soru_cevap.py       # Terminalden tek seferlik soru veya interaktif sohbet modu; kaynak gösterir
├── config.py           # Model isimleri, dizin, bağlam penceresi ve retriever ayarları (tek nokta)
├── requirements.txt    # Çalışan ortamın gerçek bağımlılık sürümleri (pip freeze --local)
├── .gitignore          # venv/, chroma_db/, web_chroma_db/, yuklenen_pdfler/, __pycache__/ hariç tutulur
├── chroma_db/          # CLI'nin vektör veritabanı (otomatik oluşur, repoya dahil edilmez)
├── web_chroma_db/      # Web arayüzünün kendi vektör veritabanı (otomatik oluşur, repoya dahil edilmez)
├── yuklenen_pdfler/    # Web arayüzünden yüklenen PDF'lerin geçici kopyaları (repoya dahil edilmez)
└── README.md
```

> ℹ️ Web arayüzü (`app.py`) ve CLI (`rag_motoru.py` / `soru_cevap.py`) **birbirinden ayrı veritabanları** kullanır (`web_chroma_db/` ve `chroma_db/`). Bu bilinçli bir tasarım kararı: web arayüzü her zaman kendi başına, önceden hazır içerik olmadan açılsın diye.

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
streamlit               # web arayüzü
```

> 💡 Bu dosya, proje için oluşturulmuş **izole bir sanal ortamda** `pip freeze --local` ile üretilmiştir. `--local` bayrağı önemlidir: sistem genelinde kurulu başka paketlerin (örn. ROS2 gibi) listeye sızmasını engeller ve sadece bu projeye ait bağımlılıkları listeler.

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
5. `config.py` içindeki ayarları kendi kurulumunuza göre kontrol edin:
   ```python
   EMBED_MODEL = "nomic-embed-text"
   LLM_MODEL = "llama3"
   DB_DIR = "./chroma_db"
   RETRIEVER_K = 2
   NUM_CTX = 8192   # modelin bağlam penceresi (token); yüksek k değerleriyle çalışırken büyütülür
   ```

---

## ▶️ Kullanım — İki Yol Var

### Yol 1: Web Arayüzü (Streamlit) — Önerilen Yöntem

Kurulumdan sonra tek komutla, tarayıcıdan PDF yükleyip sohbet edebilirsiniz:

```bash
streamlit run app.py
```

Terminalde çıkan `Local URL` (genelde `http://localhost:8501`) tarayıcıda otomatik açılır.

**Nasıl kullanılır:**

1. Sayfa ilk açıldığında henüz hiçbir PDF eklenmemiştir; sohbet kutusu görünmez, sadece "önce PDF ekleyin" mesajı vardır.
2. Sol kenar çubuğundan bir veya birden fazla PDF seçin, **"➕ Veritabanına Ekle"** butonuna basın.
3. Ekleme bittiğinde dosya adı, kaç parçaya bölündüğüyle birlikte kalıcı bir ✅ onay satırı olarak sidebar'da görünür (sayfa yenilense de kaybolmaz).
4. Artık ana ekranda sohbet kutusu açılır; sorularınızı yazabilirsiniz.
5. Her cevabın altında, hangi dosya/sayfadan geldiği (📚 Kaynaklar) gösterilir.

**Sidebar'daki diğer kontroller:**

| Kontrol | Ne işe yarar |
|---|---|
| Kaydırıcı (k) | Her soruda veritabanından kaç parça getirileceğini ayarlar. Küçük/az sayfalı PDF'lerde düşük (2-4), büyük/çok sayfalı dosyalarda daha yüksek (6-10) değerler daha isabetli cevap verir. Aşırı yüksek değerler modelin bağlam penceresini zorlayıp cevap kalitesini düşürebilir. |
| 🗑️ Veritabanını Sıfırla | Web arayüzünün kendi veritabanını (`web_chroma_db/`) ve yüklenen dosyaları tamamen temizler, sıfırdan başlamanızı sağlar. |

> ℹ️ Web arayüzü kendi `web_chroma_db/` klasörünü kullanır. Programı `Ctrl+C` ile kapatıp yeniden başlatsanız bile buradaki veriler kalıcıdır — kaybolmasını istiyorsanız yukarıdaki sıfırlama butonunu kullanmanız gerekir.

---

### Yol 2: Terminal (CLI)

Otomasyon, script içinde çağırma veya arayüz olmadan hızlı test etmek isteyenler için.

#### 1. Adım: PDF'leri Vektör Veritabanına Kaydetme

```bash
python rag_motoru.py --pdf cv.pdf
```

Birden fazla PDF'i aynı anda işlemek için dosyaları boşlukla ayırarak verin:

```bash
python rag_motoru.py --pdf cv.pdf rapor.pdf sunum.pdf
```

| Parametre | Zorunlu | Açıklama |
|---|---|---|
| `--pdf` | ✅ | İşlenecek bir veya birden fazla PDF dosyasının yolu |

> ℹ️ `chroma_db` klasörü zaten varsa script hata vermez; yeni verilen dosyaları **mevcut veritabanına ekler**. Bir dosya zaten veritabanında ise (aynı yol/isimle önceden eklenmişse) otomatik olarak atlanır, tekrar işlenmez.
>
> Veritabanını tamamen sıfırlamak isterseniz:
> ```bash
> rm -rf chroma_db      # Windows: rmdir /s /q chroma_db
> ```

#### 2. Adım: Asistana Soru Sorma

**a) Tek seferlik soru** (hafızasız, otomasyon/script içinde kullanışlı):

```bash
python soru_cevap.py -s "Bu adayın projeleri nelerdir kısaca özetler misin?"
```

| Parametre | Kısa Hali | Zorunlu | Açıklama |
|---|---|---|---|
| `--soru` | `-s` | ❌ | Tek seferlik soru (tırnak içinde). Verilmezse sohbet modu açılır |

**b) İnteraktif sohbet modu** (`-s` verilmeden çalıştırılır):

```bash
python soru_cevap.py
```

```
💬 Sohbet modu başladı. Konuşma geçmişi hatırlanacak. Çıkmak için 'q' yazın.

Sen: Bu pdf kimin CV'si?
...
Sen: Peki onun okuduğu üniversite neresi?
...
Sen: q
Görüşürüz! 👋
```

Çıkmak için `q`, `quit`, `exit` veya `çık` yazabilirsiniz (`Ctrl+C` ile de çıkılabilir).

Örnek çıktı:

```
🤖 ASİSTANIN CEVABI:
--------------------------------------------------
Adayın projeleri şunlardır:
* IoT Tabanlı Arama Kurtarma ve Keşif Aracı: ...
--------------------------------------------------

📚 Kaynaklar:
  - cv.pdf, sayfa 1
  - rapor.pdf, sayfa 3
```

---

## 🛠️ Nasıl Çalışır?

1. **Yükleme:** `PyPDFLoader` ile her PDF sayfa sayfa okunur.
2. **Bölme:** `RecursiveCharacterTextSplitter` ile metin, 1000 karakterlik ve 200 karakter üst üste binen (overlap) parçalara bölünür.
3. **Embedding:** Her parça, Ollama'nın `nomic-embed-text` modeli ile vektöre dönüştürülür.
4. **Saklama:** Vektörler `langchain-chroma` paketiyle otomatik olarak kalıcı yazılır. Veritabanı zaten varsa, yeni parçalar `add_documents` ile üstüne eklenir; her dosya için önce `source` metadata'sına bakılarak zaten işlenip işlenmediği kontrol edilir.
5. **Zincir Kurulumu (LCEL):** `create_stuff_documents_chain` ile bir cevap-üretme zinciri, `create_retrieval_chain` ile de bu zinciri retriever'a bağlayan tam RAG zinciri oluşturulur. Prompt'a eklenen `MessagesPlaceholder("chat_history")` sayesinde zincir, önceki konuşmayı da görebilir.
6. **Sorgulama:** Kullanıcının sorusu, en alakalı `k` parça ile birlikte `ChatOllama` üzerinden `llama3` modeline chat formatında (`system`/`human` rolleri) gönderilir.
7. **Dil Kontrolü:** Sistem promptu, bağlam İngilizce olsa dahi modelin **daima Türkçe** yanıt vermesini zorunlu kılar.
8. **Kaynak Çıkarımı:** Retriever'ın döndürdüğü doküman parçalarının `metadata['source']` ve `metadata['page']` alanları okunarak, cevabın altına hangi dosya/sayfadan geldiği eklenir (tekrarlar temizlenir).
9. **Sohbet Hafızası:** Her soru-cevap çifti, `HumanMessage`/`AIMessage` olarak `chat_history` listesine eklenir ve bir sonraki soruda prompt'a dahil edilir. Not: hafıza şu an sadece cevap üretiminde kullanılır; retriever'ın veritabanından hangi parçaları çekeceği hâlâ sadece o anki soruya bakar (geçmişe göre sorgu yeniden yazımı henüz yok).
10. **Konu Uyumu Kontrolü:** Sistem promptu, getirilen bağlamın sorulan konuyla örtüşüp örtüşmediğini modelin kendisine kontrol ettirir; örtüşmüyorsa model bunu açıkça belirtip yanlış konuyu doğruymuş gibi sunmaktan kaçınır. Bu, retrieval'ın kendisini iyileştirmez, sadece yanlış eşleşmelerin fark edilmesini sağlar.

---

## ❗ Sorun Giderme

| Hata Mesajı / Durum | Olası Sebep | Çözüm |
|---|---|---|
| `Belirtilen '...' dosyası bulunamadı!` | PDF yolu yanlış | Dosya yolunu ve adını kontrol edin |
| `Vektör veritabanı oluşturulamadı. Ollama servisi çalışıyor mu?` | Ollama arka planda çalışmıyor | `ollama serve` komutunu çalıştırın |
| `Vektör veritabanı bulunamadı!` | Önce `rag_motoru.py` çalıştırılmamış | Önce PDF'leri işleyip veritabanını oluşturun |
| `ModuleNotFoundError: No module named 'langchain.chains'` | LangChain 1.x kurulu; eski chain fonksiyonları `langchain_classic`'e taşındı | `pip install langchain-classic` kurup importları `langchain_classic.chains`'ten yapın |
| `AttributeError: 'Chroma' object has no attribute 'persist'` | Yeni `langchain-chroma` paketinde `.persist()` kaldırıldı (otomatik persist var) | Kodda `.persist()` çağrısını silin |
| Model konu dışı/rol yapan cevaplar veriyor | `OllamaLLM` (completion modeli) ile `ChatPromptTemplate` (chat formatı) uyumsuzluğu | `OllamaLLM` yerine `ChatOllama` kullanın |
| Türkçe soruya İngilizce cevap geliyor | Bağlam (PDF içeriği) İngilizce; model dil sinyalini bağlamdan alıyor | Sistem promptunda "daima Türkçe cevap ver" talimatını net ve vurgulu şekilde tekrarlayın |
| Belirsiz/çok kısa sorularda ("test" gibi) model kendi sistem talimatlarını tekrar ediyor veya İngilizceye kayıyor | Modelin tutunacak somut bir bağlam bulamaması | Daha spesifik/anlamlı sorular sorun |
| Model, soruyla ilgisiz bir bağlamı sanki doğruymuş gibi anlatıyor | Retriever, soruya anlamsal olarak yakın ama konu olarak farklı bir parça getirmiş; model bunu fark etmemiş | Sistem promptundaki konu uyumu kontrolü bunu büyük ölçüde engeller; hâlâ olursa `k` değerini değiştirip tekrar deneyin |
| `k` değeri yüksekken (örn. 10+) cevaplar kötüleşiyor/tutarsızlaşıyor | Modelin bağlam penceresi (`num_ctx`) aşılıp promptun bir kısmı sessizce kesiliyor | `config.py`'de `NUM_CTX` değerini büyütün (örn. 8192 veya 16384) |
| Web arayüzünde eklenen PDF, programı yeniden başlatınca da duruyor | `web_chroma_db/` diske kalıcı yazıyor; bu bilinçli bir tasarım | İstediğinizde sidebar'daki "🗑️ Veritabanını Sıfırla" butonunu kullanın |
| 288 sayfa gibi büyük PDF'lerde işlem uzun sürüyor | Her sayfa sırayla (paralel değil) embedding'e gönderiliyor | Sabırla bekleyin; ileride paralelleştirme eklenebilir |
| `fontTools is required to fully parse...` uyarısı | PDF'in kendi font kodlaması eksik meta veri içeriyor | Zararsızdır, metin okumayı etkilemez; isterseniz `pip install fonttools` ile giderebilirsiniz |
| Cevap üretimi çok yavaş | Modelin çalıştığı makinenin kaynakları yetersiz kalıyor | Daha küçük/hafif bir model deneyin |
| `requirements.txt` çok uzun/ilgisiz paketler içeriyor | `pip freeze` sistem genelindeki başka paketleri de yakalamış olabilir | Temiz bir venv içinde `pip freeze --local > requirements.txt` çalıştırın |

---

## 📌 Notlar ve Bilinen Sınırlamalar

- Bu proje tamamen **yerel** çalışır; internet bağlantısı sadece ilk kurulumda Ollama modellerini indirmek için gereklidir.
- Web arayüzü ve CLI **birbirinden ayrı veritabanları** kullanır; birinde eklediğiniz PDF diğerinde görünmez, bu bilinçli bir tasarım kararıdır.
- **Sohbet hafızası** sadece cevap üretiminde kullanılır; retrieval (hangi PDF parçalarının bulunacağı) hâlâ o anki soruya göre yapılır. Uzun sohbetlerde önceki bağlamla ilgili parçaların retriever tarafından tekrar bulunamaması mümkündür.
- **Konu uyumu kontrolü** bir güvence değil, bir azaltma önlemidir: model hâlâ, özellikle kurnaz/varsayım içeren sorularda ("X ile Y arasında bağlantı var mı?" gibi), zorlama bir bağlantı kurabilir. Kritik kullanımlarda cevapları her zaman kaynak sayfalarıyla çarpraz kontrol edin.
- Cevap kalitesi kullanılan LLM modeline (`llama3`), chunk boyutuna, `RETRIEVER_K` (veya web arayüzündeki k kaydırıcısı) ve `NUM_CTX` değerine göre değişebilir.
- Proje, LangChain'in hızlı sürüm geçişlerine (0.1 → 0.3 → 1.x) uyum sağlayacak şekilde güncel tutulmuştur; `langchain_classic` gibi paket taşımalarını takip etmek gelecekte de gerekebilir.
- Planlanan sonraki geliştirmeler: otomatik testler / GitHub Actions CI, retrieval'ın sohbet geçmişine göre sorguyu yeniden yazması (query rewriting).

---

## 📄 Lisans

Bu proje dilediğiniz gibi kullanılabilir. Bir lisans eklemek isterseniz [MIT Lisansı](https://opensource.org/licenses/MIT) önerilir.