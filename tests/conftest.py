"""
Testlerin ortak yardımcıları. Hiçbir test Ollama'ya ihtiyaç duymaz:
embedding için langchain_core'un deterministik sahte embedding'i, LLM için
sahte sohbet modeli kullanılır; Chroma ise bellekte (diske yazmadan) çalışır.
"""
import os
import sys
import uuid

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetrisi testlerde kapalı
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORNEK_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ornek.pdf")


@pytest.fixture
def sahte_embeddings():
    from langchain_core.embeddings import DeterministicFakeEmbedding
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def bellek_db(sahte_embeddings):
    """Her test için ayrı, bellekte çalışan boş bir Chroma koleksiyonu."""
    from langchain_chroma import Chroma
    db = Chroma(collection_name=f"test_{uuid.uuid4().hex}", embedding_function=sahte_embeddings)
    yield db
    db.delete_collection()


@pytest.fixture
def ornek_pdf():
    return ORNEK_PDF
