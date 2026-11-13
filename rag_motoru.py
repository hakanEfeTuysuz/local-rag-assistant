import argparse
import hashlib
import logging
import os
import sys
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHUNK_OVERLAP, CHUNK_SIZE, DB_DIR, EMBED_PARTI_BOYUTU  # MERKEZİ AYARLAR
from rag_zinciri import (
    PdfOkumaHatasi,
    RagHatasi,
    VeritabaniHatasi,
    cli_loglamasini_ayarla,
    embedding_yukle,
    vektor_db_ac,
)

logger = logging.getLogger(__name__)

# Windows'ta dosya adında kullanılamayan karakterler
_GECERSIZ_KARAKTERLER = set('<>:"/\\|?*')
_MAX_AD_BAYT = 150  # dosya sistemi sınırının (255 bayt) güvenli şekilde altında


def dosya_hash_hesapla(dosya_yolu: str) -> str:
    """Dosyanın içeriğinden SHA-256 özeti üretir (aynı içerik = aynı özet)."""
    ozet = hashlib.sha256()
    with open(dosya_yolu, "rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            ozet.update(blok)
    return ozet.hexdigest()


def guvenli_dosya_adi(isim: str) -> str:
    """
    Dışarıdan gelen (örn. tarayıcının gönderdiği) dosya adını, diske güvenle
    yazılabilecek sade bir dosya adına çevirir. '../../x.pdf' gibi klasör
    dışına çıkmaya çalışan yollar sadece 'x.pdf' olarak kalır.
    """
    ad = os.path.basename(isim.replace("\\", "/")).strip()
    ad = "".join(c for c in ad if c.isprintable() and c not in _GECERSIZ_KARAKTERLER)
    if ad.strip(". ") == "":
        raise ValueError(f"Geçersiz dosya adı: {isim!r}")

    # Çoğu dosya sistemi dosya adını 255 BAYT ile sınırlar (Türkçe harfler 2 bayt).
    # Çok uzun adlar uzantı korunarak kısaltılır, yoksa diske yazarken hata alınır.
    if len(ad.encode("utf-8")) > _MAX_AD_BAYT:
        kok, uzanti = os.path.splitext(ad)
        uzanti = uzanti[:10]
        while kok and len((kok + uzanti).encode("utf-8")) > _MAX_AD_BAYT:
            kok = kok[:-1]
        ad = kok.rstrip() + uzanti
    return ad


def pdf_hazirla(
    dosya_yolu: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, dosya_hash: str | None = None
) -> list[Document]:
    """
    PDF'i sayfa sayfa okur ve parçalara böler. Her parçanın metadata'sında
    'source' (dosyanın mutlak yolu), 'page' (0'dan başlayan sayfa no) ve
    'dosya_hash' (içerik özeti, tekrar eklemeyi önlemek için) bulunur.

    dosya_hash daha önce hesaplandıysa verilebilir; dosya ikinci kez okunmaz.
    Metni çıkarılamayan tek tük bozuk sayfalar tüm PDF'i reddettirmez, atlanır.
    """
    logger.info(f"1. Aşama: '{dosya_yolu}' yükleniyor...")

    if not os.path.isfile(dosya_yolu):
        raise FileNotFoundError(f"Belirtilen '{dosya_yolu}' dosyası bulunamadı! Lütfen yolu kontrol edin.")

    kaynak = os.path.abspath(dosya_yolu)
    ad = os.path.basename(dosya_yolu)  # hata mesajlarında sunucu yolunu değil sadece dosya adını göster
    try:
        if dosya_hash is None:
            dosya_hash = dosya_hash_hesapla(dosya_yolu)
        okuyucu = PdfReader(dosya_yolu)
        # Boş parolayla şifrelenmiş PDF'ler (sadece kopyalama/yazdırma kısıtlı) okunabilir;
        # gerçek bir parola isteyenler ise anlaşılır bir mesajla reddedilir.
        if okuyucu.is_encrypted and not okuyucu.decrypt(""):
            raise PdfOkumaHatasi(f"'{ad}' parola korumalı. PDF'in parolasını kaldırıp tekrar deneyin.")
        sayfa_sayisi = len(okuyucu.pages)
    except PdfOkumaHatasi:
        raise
    except Exception as e:
        raise PdfOkumaHatasi(f"'{ad}' PDF olarak okunamadı (bozuk veya şifreli olabilir). Detay: {e}") from e

    dokumanlar, okunamayanlar = [], []
    for i in range(sayfa_sayisi):
        try:
            metin = okuyucu.pages[i].extract_text() or ""
        except Exception:
            okunamayanlar.append(i + 1)
            continue
        dokumanlar.append(Document(
            page_content=metin,
            metadata={"source": kaynak, "page": i, "total_pages": sayfa_sayisi, "dosya_hash": dosya_hash},
        ))
    logger.info(f"-> Başarılı! PDF toplam {sayfa_sayisi} sayfa olarak okundu.\n")
    if okunamayanlar:
        logger.warning(f"⚠️  '{ad}': {len(okunamayanlar)} sayfanın metni okunamadı ve atlandı (sayfa {', '.join(map(str, okunamayanlar))}).")

    logger.info("2. Aşama: Metin parçalara bölünüyor (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    parcalar = text_splitter.split_documents(dokumanlar)
    if not parcalar:
        raise PdfOkumaHatasi(
            f"'{ad}' içinden hiç metin çıkarılamadı. Taranmış (görsel) bir PDF olabilir; "
            "bu proje OCR yapmadığı için sadece metin içeren PDF'leri işleyebilir."
        )
    logger.info(f"-> Başarılı! Metin toplam {len(parcalar)} anlamsal parçaya bölündü.\n")

    return parcalar


def dosya_zaten_islendi_mi(vektor_db, dosya_yolu: str, dosya_hash: str | None = None) -> bool:
    """
    Aynı dosyanın veritabanına daha önce eklenip eklenmediğini kontrol eder.
      - Yol karşılaştırması normalize edilir: 'cv.pdf', './cv.pdf' ve tam yol
        aynı dosya sayılır (eski sürümün kaydettiği göreli yollar da bulunur).
      - Dosya diskte varsa içerik özeti (hash) de aranır: aynı dosya farklı bir
        klasörden/isimle verilse bile tekrar eklenmez. Önceden hesaplandıysa
        dosya_hash verilebilir; dosya tekrar okunmaz.
    """
    yol_adaylari = sorted({dosya_yolu, os.path.normpath(dosya_yolu), os.path.abspath(dosya_yolu)})
    kosullar = [{"source": {"$in": yol_adaylari}}]
    if dosya_hash is None and os.path.isfile(dosya_yolu):
        dosya_hash = dosya_hash_hesapla(dosya_yolu)
    if dosya_hash:
        kosullar.append({"dosya_hash": dosya_hash})

    where = kosullar[0] if len(kosullar) == 1 else {"$or": kosullar}
    try:
        sonuc = vektor_db.get(where=where, limit=1, include=[])
    except Exception as e:
        raise VeritabaniHatasi(f"Veritabanında dosya kontrolü yapılamadı. Detay: {e}") from e
    return len(sonuc.get("ids", [])) > 0


def parca_kimlikleri(parcalar: list[Document]) -> list[str]:
    """
    Her parça için kalıcı bir kimlik üretir: '<dosya_hash>-<sıra>'. Aynı dosya
    tekrar eklenirse aynı kimlikler oluşur ve Chroma kopya oluşturmak yerine
    üzerine yazar. Hash'i olmayan parçalarda (dışarıdan gelen Document'lar)
    rastgele kimlik kullanılır.
    """
    return [
        f"{p.metadata['dosya_hash']}-{i}" if p.metadata.get("dosya_hash") else uuid.uuid4().hex
        for i, p in enumerate(parcalar)
    ]


def vektor_veritabanina_kaydet(parcalar: list[Document], vektor_db, ilerleme=None, parti_boyutu: int = EMBED_PARTI_BOYUTU):
    """
    Parçaları embedding'e çevirip veritabanına ekler (Chroma otomatik kalıcı yazar).

    - Parçalar 'parti_boyutu'luk gruplar halinde gönderilir; her gruptan sonra
      ilerleme(tamamlanan, toplam) çağrılır (büyük PDF'lerde ilerleme göstermek için).
    - Bir grup başarısız olursa bu çağrıda eklenen tüm parçalar geri silinir.
      Böylece yarım eklenmiş bir dosya sonraki denemede "zaten eklenmiş"
      sanılıp eksik kalmaz; dosya baştan tekrar eklenebilir.
    """
    toplam = len(parcalar)
    logger.info(f"3. Aşama: {toplam} parça embedding'e çevrilip veritabanına ekleniyor...")
    kimlikler = parca_kimlikleri(parcalar)
    try:
        for bas in range(0, toplam, parti_boyutu):
            son = min(bas + parti_boyutu, toplam)
            vektor_db.add_documents(parcalar[bas:son], ids=kimlikler[bas:son])
            if ilerleme is not None:
                ilerleme(son, toplam)
            if toplam > parti_boyutu:
                logger.info(f"   ... {son}/{toplam} parça eklendi")
    except Exception as e:
        try:
            vektor_db.delete(ids=kimlikler)
        except Exception as silme_hatasi:
            logger.warning(f"Yarım kalan parçalar geri silinemedi: {silme_hatasi}")
        raise VeritabaniHatasi(f"Veritabanına ekleme yapılamadı. Ollama servisi çalışıyor mu? Detay: {e}") from e
    logger.info(f"-> Başarılı! {toplam} yeni parça veritabanına eklendi.\n")
    return vektor_db


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bir veya birden fazla PDF dosyasını RAG sistemi için vektör veritabanına kaydeder.")
    parser.add_argument("--pdf", type=str, required=True, nargs="+",
                        help="İşlenecek PDF dosyalarının yolu (birden fazla dosya için aralarına boşluk koyun, örn: --pdf cv.pdf rapor.pdf)")
    args = parser.parse_args(argv)
    cli_loglamasini_ayarla()

    try:
        if os.path.exists(DB_DIR):
            logger.info(f"Mevcut veritabanı bulundu: '{DB_DIR}'. Yeni dosyalar buna eklenecek.\n")
        else:
            logger.info(f"Yeni vektör veritabanı oluşturuluyor: '{DB_DIR}'\n")
        vektor_db = vektor_db_ac(DB_DIR, embedding_yukle())
    except RagHatasi as hata:
        print(f"\n❌ SİSTEM HATASI: {hata}", file=sys.stderr)
        return 1

    # Bir dosyadaki hata diğerlerinin işlenmesini engellemez; sonda özetlenir.
    basarisizlar = []
    for dosya_yolu in args.pdf:
        try:
            # İçerik özeti bir kez hesaplanır; hem tekrar kontrolünde hem parçalarda kullanılır
            dosya_hash = dosya_hash_hesapla(dosya_yolu) if os.path.isfile(dosya_yolu) else None
            if dosya_zaten_islendi_mi(vektor_db, dosya_yolu, dosya_hash):
                logger.info(f"⏭️  '{dosya_yolu}' zaten veritabanında var, atlanıyor.\n")
                continue
            parcalar = pdf_hazirla(dosya_yolu, dosya_hash=dosya_hash)
            vektor_veritabanina_kaydet(parcalar, vektor_db)
        except (RagHatasi, OSError) as hata:  # OSError: dosya yok / okunamıyor
            print(f"\n❌ HATA ('{dosya_yolu}'): {hata}\n", file=sys.stderr)
            basarisizlar.append(dosya_yolu)

    if basarisizlar:
        print(f"⚠️  {len(basarisizlar)} dosya işlenemedi: {', '.join(basarisizlar)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
