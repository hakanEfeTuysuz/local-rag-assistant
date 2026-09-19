#!/usr/bin/env bash
# ROS2 gibi sistem genelindeki bazı kurulumlar PYTHONPATH'e kendi paketlerini
# ekleyebiliyor; bu da pytest'in başlangıçta o paketleri "eklenti" (plugin)
# sanıp yüklemeye çalışmasına ve hataya düşmesine sebep olabiliyor.
# Bu satır, pytest'in otomatik eklenti taramasını tamamen kapatarak
# projeyi kimin, hangi makinede klonladığından bağımsız hale getirir.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

pytest tests/ -v "$@"