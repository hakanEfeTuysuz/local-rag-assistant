from rag_motoru import dosya_zaten_islendi_mi


class SahteVektorDB:
    """
    dosya_zaten_islendi_mi fonksiyonu sadece vektor_db.get(where=...) çağırıp
    dönen sözlükteki 'ids' listesine bakıyor. Gerçek bir Chroma kurmaya
    gerek yok, sadece .get() metodunu taklit eden bir sahte nesne yeterli.
    """
    def __init__(self, kayitli_id_listesi):
        self._kayitli_id_listesi = kayitli_id_listesi

    def get(self, where=None):
        if self._kayitli_id_listesi:
            return {"ids": self._kayitli_id_listesi}
        return {"ids": []}


def test_dosya_veritabaninda_varsa_true_doner():
    sahte_db = SahteVektorDB(kayitli_id_listesi=["id1", "id2"])
    assert dosya_zaten_islendi_mi(sahte_db, "cv.pdf") is True


def test_dosya_veritabaninda_yoksa_false_doner():
    sahte_db = SahteVektorDB(kayitli_id_listesi=[])
    assert dosya_zaten_islendi_mi(sahte_db, "cv.pdf") is False