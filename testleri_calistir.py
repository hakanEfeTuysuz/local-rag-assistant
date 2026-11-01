#!/usr/bin/env python3
"""
testleri_calistir.sh'in platform bağımsız (Windows dahil) karşılığı.

ROS2 gibi sistem genelindeki bazı kurulumlar PYTHONPATH'e kendi paketlerini
ekleyebiliyor; bu da pytest'in başlangıçta o paketleri "eklenti" (plugin)
sanıp yüklemeye çalışmasına ve hataya düşmesine sebep olabiliyor.
Aşağıdaki satır, pytest'in otomatik eklenti taramasını tamamen kapatarak
projeyi kimin, hangi işletim sisteminde çalıştırdığından bağımsız hale getirir.

Önemli: bu satır, 'import pytest' satırından ÖNCE çalışmalı.
"""
import os

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

import sys

import pytest

if __name__ == "__main__":
    ek_argumanlar = sys.argv[1:]
    sys.exit(pytest.main(["tests/", "-v"] + ek_argumanlar))
