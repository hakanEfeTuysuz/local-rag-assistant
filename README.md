# 📚 Yerel RAG Asistanı (PDF + Ollama + LangChain)

[![Testler](https://github.com/hakanEfeTuysuz/local-rag-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/hakanEfeTuysuz/local-rag-assistant/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Lisans: MIT](https://img.shields.io/badge/lisans-MIT-green)](LICENSE)

PDF dosyalarını okuyup vektör veritabanına kaydeden ve ardından bu veritabanı üzerinden **tamamen yerel** (internet gerektirmeyen) bir LLM ile soru-cevap yapabilen, LCEL mimarisiyle inşa edilmiş bir RAG (Retrieval Augmented Generation) sistemi. Tarayıcıda açılan bir **Streamlit web arayüzü** ile PDF yükleyip anında sohbet edebilir, isterseniz terminal (CLI) üzerinden de kullanabilirsiniz.

Sistem, embedding ve dil modeli için [Ollama](https://ollama.com) kullanır; bu sayede verileriniz hiçbir zaman bilgisayarınızdan dışarı çıkmaz.

---

## 🎯 Proje Amacı / Motivasyon

Bu proje, bulut API'lerine (OpenAI vb.) bağımlı kalmadan, tamamen yerel çalışan ve veri gizliliğini merkeze alan gerçek dünya AI çözümleri geliştirmek amacıyla bir **Proof of Concept (PoC)** olarak hazırlanmıştır. "Tutorial hell"den çıkıp, RAG mimarisini sıfırdan kurma ve modern LangChain sürümlerine (1.x) taşıma pratiğidir.

---

## 🚀 Özellikler

- **Streamlit web arayüzü**: tarayıcıdan sürükle-bırak PDF yükleme, ayarlanabilir arama derinliği (k), kalıcı oturum ve **akış halinde (token token) gelen cevaplar** — kurulumdan sonra tek komutla açılır
- **Modern ve sade arayüz**: açık/koyu temaya otomatik uyum, telefonda da düzgün görünüm, ilk açılışta 3 adımlı rehber, tek tıkla sorulabilen örnek sorular, cevap süresi ve kaynak rozetleri, onay isteyen veritabanı sıfırlama ve "Sohbeti Temizle" butonu. Yazı tipleri dahil hiçbir şey internetten yüklenmez
- PDF dosyasını otomatik olarak okuma ve sayfalara ayırma
- Metni anlamsal parçalara bölme (chunking)
- `nomic-embed-text` modeli ile embedding oluşturma
- Embedding'leri **ChromaDB** içinde kalıcı olarak saklama (otomatik persist)
- **Çoklu PDF desteği**: veritabanı zaten varsa yeni dosyalar üstüne **eklenir** (append); aynı dosya tekrar verilirse — farklı bir yoldan veya isimle bile (içerik özeti/SHA-256 ile tanınır) — otomatik atlanır
- **Kaynak gösterme**: her cevabın altında, bilginin hangi PDF dosyasından ve hangi sayfalardan geldiği listelenir (`kitap.pdf, sayfa 1, 3, 4`)
- **Kaynak metinleri (web)**: cevabın altındaki açılır **📖 Kaynak metinleri** bölümünde, modelin gördüğü PDF parçaları sayfa bilgisiyle gösterilir; böylece cevabı belgeyle karşılaştırıp kendiniz doğrulayabilirsiniz
- **Sohbet hafızası + soru yeniden yazma**: konuşma geçmişi hatırlanır; "onun", "bunun" gibi zamirler içeren sorular, arama yapılmadan önce geçmişe göre kendi başına anlamlı bir soruya dönüştürülür (query rewriting), böylece veritabanında doğru parçalar bulunur. Geçmiş, bağlam penceresini taşırmasın diye son N mesajla sınırlanır
- **Konu uyumu kontrolü**: getirilen bağlam, sorulan konuyla örtüşmüyorsa model bunu doğruymuş gibi sunmak yerine açıkça belirtir
- **Otomatik testler + CI**: PDF işleme, tekrar-ekleme kontrolü, RAG zinciri (sahte model + bellekteki Chroma ile), sohbet geçmişi ve ayarlar ve web arayüzü için 77 pytest testi; her `git push`'ta GitHub Actions üzerinde Ubuntu ve Windows'ta çalışır, kod ayrıca `ruff` ile denetlenir. Testler Ollama gerektirmez
- **LCEL (LangChain Expression Language)** mimarisiyle kurulmuş modern RAG zinciri
- `ChatOllama` ile chat-formatlı prompt kullanımı (sistem/kullanıcı rolleri)
- Sistem promptu ile **her koşulda Türkçe yanıt** garantisi — bağlam İngilizce olsa bile
- Model, dizin, chunk boyutu, bağlam penceresi, retriever ve geçmiş ayarlarının `config.py` üzerinden merkezi yönetimi; her ayar koda dokunmadan **ortam değişkeniyle** de değiştirilebilir
- Web arayüzü ve CLI **aynı RAG çekirdeğini** (`rag_zinciri.py`) kullanır; prompt ve zincir tek bir yerde tanımlıdır
- Anlaşılır Türkçe hata mesajları, aşamaya özel hata sınıfları (`PdfOkumaHatasi`, `VeritabaniHatasi`, `ModelHatasi`), `raise ... from e` ile korunmuş hata zinciri; CLI hata durumunda `1` çıkış koduyla biter
- Web arayüzünden yüklenen dosya adları temizlenir (`../../` gibi klasör dışına yazma denemeleri engellenir, çok uzun adlar kısaltılır)
- **Gizlilik odaklı varsayılanlar**: web arayüzü sadece bu bilgisayardan erişilebilir (aynı ağdaki başkalarına açılmaz) ve Streamlit'in kullanım istatistiği gönderimi kapalıdır (`.streamlit/config.toml`)

---

## 🗂️ Proje Yapısı

```
.
├── app.py                    # Streamlit web arayüzü: PDF yükleme + sohbet, tarayıcı üzerinden
├── rag_zinciri.py            # Ortak RAG çekirdeği: modeller, promptlar, zincir, geçmiş kırpma, hata sınıfları
├── rag_motoru.py             # PDF'leri işleyip vektör veritabanını oluşturan/güncelleyen script (CLI)
├── soru_cevap.py             # Terminalden tek seferlik soru veya interaktif sohbet modu; kaynak gösterir
├── config.py                 # Tüm ayarlar (tek nokta); ortam değişkenleriyle değiştirilebilir
├── tests/                    # Ollama'ya bağımlı olmayan pytest testleri
│   ├── conftest.py           # Ortak test yardımcıları (sahte embedding, bellekteki Chroma)
│   ├── test_app.py           # Web arayüzünün uçtan uca testleri (Streamlit AppTest + sahte model)
│   ├── test_config.py
│   ├── test_rag_motoru.py
│   ├── test_rag_zinciri.py
│   └── test_soru_cevap.py
├── testleri_calistir.sh      # Testleri, sistem kaynaklı plugin sorunlarından bağımsız çalıştıran script (Linux/Mac)
├── testleri_calistir.py      # Aynı işi yapan, Windows dahil her platformda çalışan Python karşılığı
├── pytest.ini                # pytest yapılandırması
├── .github/workflows/
│   └── tests.yml             # Her push'ta testleri otomatik çalıştıran GitHub Actions tanımı
├── requirements.in           # Projenin doğrudan kullandığı paketler (elle düzenlenen kısa liste)
├── requirements.txt          # Tüm bağımlılıkların test edilmiş, sabitlenmiş sürümleri (kurulumda bunu kullanın)
├── ruff.toml                 # Kod denetleyici (linter) ayarları
├── .streamlit/config.toml    # Web arayüzü ayarları: tema (açık/koyu), sadece yerel erişim, istatistik gönderimi kapalı
├── .gitignore                # venv/, chroma_db/, web_chroma_db/, yuklenen_pdfler/, __pycache__/ hariç tutulur
├── chroma_db/                # CLI'nin vektör veritabanı (otomatik oluşur, repoya dahil edilmez)
├── web_chroma_db/            # Web arayüzünün kendi vektör veritabanı (otomatik oluşur, repoya dahil edilmez)
├── yuklenen_pdfler/          # Web arayüzünden yüklenen PDF'lerin geçici kopyaları (repoya dahil edilmez)
└── README.md
```

> ℹ️ Web arayüzü (`app.py`) ve CLI (`rag_motoru.py` / `soru_cevap.py`) **birbirinden ayrı veritabanları** kullanır (`web_chroma_db/` ve `chroma_db/`). Bu bilinçli bir tasarım kararı: web arayüzü her zaman kendi başına, önceden hazır içerik olmadan açılsın diye.

---

## ⚙️ Gereksinimler

- Python 3.10+
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

Projenin **doğrudan** kullandığı paketler `requirements.in` dosyasında listelenir:

```txt
langchain-core
langchain-classic        # create_retrieval_chain, create_history_aware_retriever, create_stuff_documents_chain
langchain-text-splitters # RecursiveCharacterTextSplitter
langchain-ollama         # OllamaEmbeddings, ChatOllama
langchain-chroma         # Chroma (otomatik persist eden yeni nesil paket)
chromadb
pypdf                    # PDF okuma
streamlit                # web arayüzü
pytest                   # testler
```

> 💡 `requirements.txt` ise bu paketlerin **tüm alt bağımlılıklarıyla birlikte** test edilmiş sürümlere sabitlenmiş halidir (uzun görünmesi normaldir; örn. `kubernetes`, `chromadb`'nin bir alt bağımlılığıdır). İzole bir sanal ortamda `pip freeze --local` ile üretilmiştir; `--local` bayrağı sistem genelindeki paketlerin (örn. ROS2) listeye sızmasını engeller. Yeni bir paket eklerken önce `requirements.in`'e yazın, kurun, sonra `requirements.txt`'yi yeniden üretin.

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
5. `config.py` içindeki varsayılan ayarları kontrol edin. Her ayar, koda dokunmadan aynı isimli ortam değişkeniyle de değiştirilebilir:

   | Ayar | Varsayılan | Ortam değişkeni | Açıklama |
   |---|---|---|---|
   | `EMBED_MODEL` | `nomic-embed-text` | `RAG_EMBED_MODEL` | Embedding modeli |
   | `LLM_MODEL` | `llama3` | `RAG_LLM_MODEL` | Sohbet modeli |
   | `DB_DIR` | `chroma_db` | `RAG_DB_DIR` | CLI'nin veritabanı klasörü |
   | `WEB_DB_DIR` | `web_chroma_db` | `RAG_WEB_DB_DIR` | Web arayüzünün veritabanı klasörü |
   | `UPLOAD_KLASORU` | `yuklenen_pdfler` | `RAG_UPLOAD_KLASORU` | Web'den yüklenen PDF'lerin kopyaları |
   | `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | Parça boyutu ve örtüşmesi (karakter) |
   | `RETRIEVER_K` | `4` | `RAG_RETRIEVER_K` | Her soruda getirilecek parça sayısı (web'de kaydırıcının varsayılanı) |
   | `NUM_CTX` | `8192` | `RAG_NUM_CTX` | Modelin bağlam penceresi (token); yüksek k değerlerinde büyütün |
   | `GECMIS_MAX_MESAJ` | `10` | `RAG_GECMIS_MAX_MESAJ` | Modele gönderilecek en fazla geçmiş mesaj (1 soru + 1 cevap = 2) |
   | `EMBED_PARTI_BOYUTU` | `64` | `RAG_EMBED_PARTI_BOYUTU` | Embedding'e tek seferde gönderilen parça sayısı (ilerleme bu adımlarla gösterilir) |

   Ollama başka bir makinede/portta çalışıyorsa, Ollama'nın kendi `OLLAMA_HOST` ortam değişkenini kullanabilirsiniz (örn. `OLLAMA_HOST=http://192.168.1.10:11434`).

   Göreli klasör yolları, komutun nereden çalıştırıldığından bağımsız olarak **proje klasörüne** göre çözülür. Örnek:
   ```bash
   RAG_LLM_MODEL=mistral RAG_RETRIEVER_K=6 streamlit run app.py
   ```

---

## ▶️ Kullanım — İki Yol Var

### Yol 1: Web Arayüzü (Streamlit) — Önerilen Yöntem

Kurulumdan sonra tek komutla, tarayıcıdan PDF yükleyip sohbet edebilirsiniz:

```bash
streamlit run app.py
```

Terminalde çıkan adres (genelde `http://localhost:8501`) tarayıcıda otomatik açılır. Komutu **proje klasöründen** çalıştırın ki `.streamlit/config.toml` ayarları devreye girsin.

> 🔒 Uygulama varsayılan olarak **sadece bu bilgisayardan** açılabilir. Aynı ağdaki başka bir cihazdan (örn. telefonunuzdan) bilerek erişmek isterseniz: `streamlit run app.py --server.address 0.0.0.0` — bu durumda ağınızdaki herkesin belgelerinize soru sorabileceğini unutmayın.

**Nasıl kullanılır:**

1. Sayfa ilk açıldığında henüz hiçbir PDF eklenmemiştir; sohbet kutusu yerine 3 adımlı kısa bir rehber görünür.
2. Sol kenar çubuğundan bir veya birden fazla PDF seçin, **"Veritabanına Ekle"** butonuna basın (telefonda kenar çubuğu sol üstteki `>>` ile açılır).
3. Ekleme bittiğinde dosya, kaç parçaya bölündüğüyle birlikte kenar çubuğundaki **Belgeler** listesinde görünür (sayfa yenilense de kaybolmaz) ve seçim kutusu temizlenir. Aynı içerikteki bir dosya tekrar yüklenirse eklenmez ve bilgi mesajı gösterilir; **aynı isimde ama farklı içerikte** bir dosya yüklenirse karışıklık olmasın diye eklenmez, yeniden adlandırmanız istenir. Eklenemeyen dosyalar (bozuk/şifreli/taranmış PDF) için hata mesajı gösterilir.
4. Artık ana ekranda sohbet kutusu açılır; sorunuzu yazabilir veya **örnek sorulardan** birine tıklayabilirsiniz.
5. Cevap, model ürettikçe akış halinde ekrana yazılır; altında hangi dosya/sayfalardan geldiği (📄 kaynak rozetleri) ve cevabın kaç saniyede üretildiği gösterilir.
6. Cevabın altındaki **Kaynak metinleri** bölümünü açarak modelin okuduğu PDF parçalarını görebilirsiniz. Model bir bilgiyi yanlış aktarırsa (küçük yerel modellerde olabilir) doğrusunu burada hemen görürsünüz. Kaynaklar ve metinler sohbet geçmişindeki eski cevaplarda da kalıcıdır.

**Sidebar'daki diğer kontroller:**

| Kontrol | Ne işe yarar |
|---|---|
| Ayarlar → kaydırıcı (k) | Her soruda veritabanından kaç parça getirileceğini ayarlar. Küçük/az sayfalı PDF'lerde düşük (2-4), büyük/çok sayfalı dosyalarda daha yüksek (6-10) değerler daha isabetli cevap verir. Aşırı yüksek değerler modelin bağlam penceresini zorlayıp cevap kalitesini düşürebilir. |
| Sohbeti Temizle | Sadece sohbet geçmişini siler; eklenen belgeler kalır. |
| Veritabanını Sıfırla | Onay istedikten sonra web arayüzünün kendi veritabanını (`web_chroma_db/`) ve yüklenen dosyaları tamamen temizler. |

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

> ℹ️ `chroma_db` klasörü zaten varsa script hata vermez; yeni verilen dosyaları **mevcut veritabanına ekler**. Bir dosya zaten veritabanında ise otomatik olarak atlanır, tekrar işlenmez. `cv.pdf`, `./cv.pdf` ve tam yol aynı dosya sayılır; içerik özeti de karşılaştırıldığı için aynı dosyanın başka bir klasördeki kopyası da atlanır.
>
> Birden fazla dosya verildiğinde birindeki hata (örn. dosya bulunamadı, bozuk PDF) diğerlerinin işlenmesini engellemez; sonda başarısız dosyalar listelenir ve script `1` çıkış koduyla biter (otomasyon/script içinde hatayı yakalayabilmeniz için). Her şey yolundaysa çıkış kodu `0`'dır.
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

Çıkmak için `q`, `quit`, `exit` veya `çık` yazabilirsiniz (`Ctrl+C` ile de çıkılabilir). Sohbet sırasında tek bir soruda hata olursa (örn. Ollama bir anlığına yanıt vermezse) sohbet kapanmaz ve geçmiş kaybolmaz; hata gösterilir ve yeni soru sorabilirsiniz.

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

## 🧪 Testler ve CI

Testlerin hiçbiri Ollama gerektirmez: embedding için `langchain_core`'un deterministik sahte embedding'i, LLM için sahte sohbet modeli (`FakeListChatModel`) kullanılır, Chroma ise bellekte çalışır. Böylece CI'da bile **tüm RAG zinciri uçtan uca** test edilebilir. Kapsanan başlıca konular:

| Dosya | Neyi test eder |
|---|---|
| `test_rag_motoru.py` | `ornek.pdf` ile PDF okuma/parçalama ve metadata, bozuk/metinsiz/olmayan PDF hataları, tekrar-ekleme kontrolü (göreli/mutlak yol, aynı içerikli kopya, eski sürüm kayıtları), gruplar halinde ekleme + ilerleme, **yarıda kalan eklemenin geri alınması**, dosya adı temizleme, CLI çıkış kodları |
| `test_rag_zinciri.py` | Kaynak listeleme, geçmiş kırpma, ilk soruda doğrudan arama, takip sorusunda **sorunun yeniden yazılıp öyle aranması**, akış (stream) çıktısı, gerçek Chroma ile zincir |
| `test_soru_cevap.py` | Hata sonrası sohbetin devam etmesi, başarısız sorunun geçmişe yazılmaması, kırpılmış geçmişin gönderilmesi, veritabanı yokken anlaşılır hata |
| `test_config.py` | Varsayılanlar, ortam değişkeniyle ayar ezme, geçersiz değerlerin anlaşılır hatası |
| `test_app.py` | Web arayüzü uçtan uca: PDF yükleme, soru-cevap, kaynaklar ve kaynak metinleri, boş soru, kötü niyetli/çok uzun dosya adları, tekrar ekleme, sıfırlama |

**Testleri yerel olarak çalıştırmak için:**

```bash
./testleri_calistir.sh
```

> 🪟 **Windows kullanıcıları:** Yukarıdaki `.sh` script'i doğrudan Windows CMD/PowerShell'de çalışmaz (Git Bash veya WSL gerektirir). Onun yerine, aynı işi yapan platform bağımsız Python karşılığını kullanabilirsiniz:
> ```bash
> python testleri_calistir.py
> ```

Her iki script de aynı korumayı sağlar: pytest'in başlangıçta sistem genelindeki paketleri (örn. ROS2 kurulu makinelerde olduğu gibi) otomatik eklenti (plugin) olarak yüklemeye çalışmasını engeller, aksi halde ilgisiz bir hata alınabilir.

Beklenen çıktının sonu:

```
============================== 77 passed in 6.3s ==============================
```

**Kod denetimi (lint):** `pip install ruff` ile kurup `ruff check .` çalıştırabilirsiniz (ayarlar `ruff.toml` içinde).

**CI (GitHub Actions):** `.github/workflows/tests.yml` sayesinde, bu depoya her `git push` yapıldığında kod önce `ruff` ile denetlenir, ardından testler GitHub'ın kendi sunucularında (sıfırdan, temiz **Ubuntu ve Windows** ortamlarında) otomatik olarak çalışır. Sonuç, repo'nun **Actions** sekmesinde ✅ (geçti) veya ❌ (başarısız) olarak görülebilir — bu sayede bir değişiklik kritik bir fonksiyonu bozarsa, kod canlıya/kullanıma geçmeden önce fark edilir.

---

## 🛠️ Nasıl Çalışır?

1. **Yükleme:** `pypdf` ile her PDF sayfa sayfa okunur. Metin içermeyen (taranmış) veya bozuk PDF'ler anlaşılır bir hatayla reddedilir.
2. **Bölme:** `RecursiveCharacterTextSplitter` ile metin, varsayılan olarak 1000 karakterlik ve 200 karakter üst üste binen (overlap) parçalara bölünür (`CHUNK_SIZE` / `CHUNK_OVERLAP`).
3. **Embedding:** Her parça, Ollama'nın `nomic-embed-text` modeli ile vektöre dönüştürülür.
4. **Saklama:** Vektörler `langchain-chroma` paketiyle otomatik olarak kalıcı yazılır. Her parçanın metadata'sında dosyanın tam yolu (`source`), sayfa numarası (`page`) ve içerik özeti (`dosya_hash`) saklanır. Her dosya için önce bu alanlara bakılarak zaten işlenip işlenmediği kontrol edilir. Parçalar `EMBED_PARTI_BOYUTU`'luk gruplar halinde eklenir ve ilerleme gösterilir (web'de ilerleme çubuğu, CLI'da `12/40 parça eklendi` satırları). Ekleme yarıda kesilirse (örn. Ollama çökerse) o dosyanın eklenen parçaları geri silinir; böylece yarım kalmış bir dosya "zaten eklenmiş" sanılmaz, tekrar eklenebilir. Parça kimlikleri içerik özetinden türetildiği için aynı dosya iki kez eklense bile kopya oluşmaz.
5. **Zincir Kurulumu (LCEL, `rag_zinciri.py`):** `create_history_aware_retriever` ile sohbet geçmişine duyarlı bir retriever, `create_stuff_documents_chain` ile bir cevap-üretme zinciri, `create_retrieval_chain` ile de ikisini birleştiren tam RAG zinciri oluşturulur. Web arayüzü ve CLI aynı fonksiyonu kullanır.
6. **Sorgulama:** Kullanıcının sorusu, en alakalı `k` parça ile birlikte `ChatOllama` üzerinden `llama3` modeline chat formatında (`system`/`human` rolleri) gönderilir.
7. **Dil Kontrolü:** Sistem promptu, bağlam İngilizce olsa dahi modelin **daima Türkçe** yanıt vermesini zorunlu kılar.
8. **Kaynak Çıkarımı:** Retriever'ın döndürdüğü doküman parçalarının `metadata['source']` ve `metadata['page']` alanları okunarak, cevabın altına hangi dosya/sayfadan geldiği eklenir (tekrarlar temizlenir).
9. **Sohbet Hafızası ve Soru Yeniden Yazma:** Her soru-cevap çifti, `HumanMessage`/`AIMessage` olarak `chat_history` listesine eklenir. Geçmiş varsa, soru önce modele "geçmişi görmeyen birinin de anlayacağı tek bir soruya" dönüştürülür (örn. *"Onun üniversitesi neresi?"* → *"Hakan Efe Tüysüz'ün okuduğu üniversite neresi?"*) ve veritabanında bu soru aranır. İlk soruda bu adım atlanır, ek model çağrısı yapılmaz. Modele giden geçmiş son `GECMIS_MAX_MESAJ` mesajla sınırlanır.
10. **Konu Uyumu Kontrolü:** Sistem promptu, getirilen bağlamın sorulan konuyla örtüşüp örtüşmediğini modelin kendisine kontrol ettirir; örtüşmüyorsa model bunu açıkça belirtip yanlış konuyu doğruymuş gibi sunmaktan kaçınır. Bu, retrieval'ın kendisini iyileştirmez, sadece yanlış eşleşmelerin fark edilmesini sağlar.

---

## ❗ Sorun Giderme

| Hata Mesajı / Durum | Olası Sebep | Çözüm |
|---|---|---|
| `Belirtilen '...' dosyası bulunamadı!` | PDF yolu yanlış | Dosya yolunu ve adını kontrol edin |
| `'...' modeli yüklenemedi. Ollama çalışıyor mu ...?` | Ollama arka planda çalışmıyor veya model indirilmemiş | `ollama serve` çalıştırın, mesajdaki `ollama pull ...` komutuyla modeli indirin; web arayüzünde ardından sayfayı yenileyin |
| `Vektör veritabanı bulunamadı!` / `veritabanı boş!` | Önce `rag_motoru.py` çalıştırılmamış veya hiç PDF eklenmemiş | Önce PDF'leri işleyip veritabanını oluşturun |
| `... içinden hiç metin çıkarılamadı. Taranmış (görsel) bir PDF olabilir` | PDF sadece görüntüden oluşuyor; proje OCR yapmıyor | PDF'i önce bir OCR aracıyla (örn. `ocrmypdf`) metin içeren hale getirin |
| `... parola korumalı` | PDF açılış parolası istiyor | PDF'in parolasını kaldırıp (örn. PDF okuyucunuzda "parolasız kaydet") tekrar deneyin. Sadece kopyalama/yazdırma kısıtlı PDF'ler sorunsuz okunur |
| `... PDF olarak okunamadı (bozuk veya şifreli olabilir)` | Dosya bozuk veya PDF değil | Dosyayı bir PDF okuyucuyla açıp kontrol edin |
| Web arayüzüne başka bir cihazdan bağlanılamıyor | Güvenlik için uygulama sadece `localhost`'u dinler | Bilerek açmak için `streamlit run app.py --server.address 0.0.0.0` (ağdaki herkes erişebilir) |
| Web'de `'...' adında, içeriği farklı bir dosya zaten ekli` | Aynı isimli farklı bir dosya daha önce eklenmiş | Yeni dosyayı yeniden adlandırıp tekrar yükleyin |
| `ModuleNotFoundError: No module named 'langchain.chains'` | LangChain 1.x kurulu; eski chain fonksiyonları `langchain_classic`'e taşındı | `pip install langchain-classic` kurup importları `langchain_classic.chains`'ten yapın |
| `AttributeError: 'Chroma' object has no attribute 'persist'` | Yeni `langchain-chroma` paketinde `.persist()` kaldırıldı (otomatik persist var) | Kodda `.persist()` çağrısını silin |
| Model konu dışı/rol yapan cevaplar veriyor | `OllamaLLM` (completion modeli) ile `ChatPromptTemplate` (chat formatı) uyumsuzluğu | `OllamaLLM` yerine `ChatOllama` kullanın |
| Türkçe soruya İngilizce cevap geliyor | Bağlam (PDF içeriği) İngilizce; model dil sinyalini bağlamdan alıyor | Sistem promptunda "daima Türkçe cevap ver" talimatını net ve vurgulu şekilde tekrarlayın |
| Belirsiz/çok kısa sorularda ("test" gibi) model kendi sistem talimatlarını tekrar ediyor veya İngilizceye kayıyor | Modelin tutunacak somut bir bağlam bulamaması | Daha spesifik/anlamlı sorular sorun |
| Model, soruyla ilgisiz bir bağlamı sanki doğruymuş gibi anlatıyor | Retriever, soruya anlamsal olarak yakın ama konu olarak farklı bir parça getirmiş; model bunu fark etmemiş | Sistem promptundaki konu uyumu kontrolü bunu büyük ölçüde engeller; hâlâ olursa `k` değerini değiştirip tekrar deneyin |
| `k` değeri yüksekken (örn. 10+) cevaplar kötüleşiyor/tutarsızlaşıyor | Modelin bağlam penceresi (`num_ctx`) aşılıp promptun bir kısmı sessizce kesiliyor | `config.py`'de `NUM_CTX` değerini büyütün (örn. 8192 veya 16384) |
| Web arayüzünde eklenen PDF, programı yeniden başlatınca da duruyor | `web_chroma_db/` diske kalıcı yazıyor; bu bilinçli bir tasarım | İstediğinizde sidebar'daki "🗑️ Veritabanını Sıfırla" butonunu kullanın |
| 288 sayfa gibi büyük PDF'lerde işlem uzun sürüyor | Embedding yerel donanımda, gruplar halinde sırayla hesaplanıyor | İlerleme çubuğu/satırları üzerinden takip edip bekleyin; işlem yarıda kesilirse dosyayı tekrar eklemeniz yeterli |
| `fontTools is required to fully parse...` uyarısı | PDF'in kendi font kodlaması eksik meta veri içeriyor | Zararsızdır, metin okumayı etkilemez; isterseniz `pip install fonttools` ile giderebilirsiniz |
| Cevap üretimi çok yavaş | Modelin çalıştığı makinenin kaynakları yetersiz kalıyor | Daha küçük/hafif bir model deneyin |
| `requirements.txt` çok uzun/ilgisiz paketler içeriyor | `pip freeze` sistem genelindeki başka paketleri de yakalamış olabilir | Temiz bir venv içinde `pip freeze --local > requirements.txt` çalıştırın |
| `pytest` çalıştırınca `PluginValidationError` veya `ModuleNotFoundError: No module named 'lark'` gibi ilgisiz hatalar | pytest, `PYTHONPATH`'e sızmış sistem paketlerini (örn. ROS2) eklenti sanıp yüklemeye çalışıyor | Doğrudan `pytest` yerine `./testleri_calistir.sh` (veya Windows'ta `python testleri_calistir.py`) kullanın |

---

## 📌 Notlar ve Bilinen Sınırlamalar

- Bu proje tamamen **yerel** çalışır; internet bağlantısı sadece ilk kurulumda Ollama modellerini indirmek için gereklidir.
- Web arayüzü ve CLI **birbirinden ayrı veritabanları** kullanır; birinde eklediğiniz PDF diğerinde görünmez, bu bilinçli bir tasarım kararıdır.
- **Soru yeniden yazma** modelin kendisi tarafından yapılır; çoğu zaman doğru çalışsa da küçük modellerde bazen soruyu gereğinden fazla değiştirebilir. Ayrıca geçmiş olan her soruda ek bir model çağrısı yapıldığı için takip soruları biraz daha yavaştır.
- Modele sadece son `GECMIS_MAX_MESAJ` mesaj gönderilir; çok uzun sohbetlerde en eski konuşmalar "unutulur" (bu, bağlam penceresinin taşmaması için bilinçli bir sınırdır).
- **OCR yoktur**: taranmış/görsel PDF'lerden metin çıkarılamaz, bu dosyalar anlaşılır bir hatayla reddedilir.
- **Konu uyumu kontrolü** bir güvence değil, bir azaltma önlemidir: model hâlâ, özellikle kurnaz/varsayım içeren sorularda ("X ile Y arasında bağlantı var mı?" gibi), zorlama bir bağlantı kurabilir. Kritik kullanımlarda cevapları her zaman kaynak sayfalarıyla çarpraz kontrol edin.
- **Testler** zincirin tamamını sahte embedding ve sahte LLM ile test eder; gerçek modellerin **cevap kalitesi** otomatik test edilmez (bunun için Ollama'nın CI ortamında çalışması gerekirdi).
- Cevap kalitesi kullanılan LLM modeline (`llama3`), chunk boyutuna, `RETRIEVER_K` (veya web arayüzündeki k kaydırıcısı) ve `NUM_CTX` değerine göre değişebilir.
- Proje, LangChain'in hızlı sürüm geçişlerine (0.1 → 0.3 → 1.x) uyum sağlayacak şekilde güncel tutulmuştur; `langchain_classic` gibi paket taşımalarını takip etmek gelecekte de gerekebilir.
- **Arama tamamen anlamsaldır (vektör benzerliği)**: özel isim, kod, madde numarası gibi birebir eşleşme gereken sorularda, ilgili cümle uzun ve konusu farklı bir paragrafın içindeyse bulunamayabilir. 300 sayfalık sentetik bir testte, rastgele metin arasına gizlenmiş tek bir cümle bu yüzden bulunamadı (model uydurmadı, bulamadığını söyledi). Böyle durumlarda soruyu belgedeki ifadelere yakın kurmak veya `k` değerini artırmak yardımcı olur.
- Planlanan sonraki geliştirmeler: anlamsal arama + kelime araması (hibrit arama), OCR desteği, cevap kalitesini ölçen küçük bir değerlendirme (eval) seti.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.