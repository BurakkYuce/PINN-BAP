"""Faz 3 doğrulama — parametrik PINN'i farklı (B, t_f, h0, n) noktalarında test (İSKELET).

Amaç: tek bir eğitilmiş parametrik model ile parametre uzayındaki birkaç noktada
çözüm üretip değerlendirmek. Sürtünmesiz idealize bir nokta için Stoker analitik
çözümü hâlâ referans olarak kullanılabilir; sürtünmeli/gerçekçi noktalarda
karşılaştırma niteldir (veya FLO-2D/HEC-RAS gibi araçların çıktısıyla yapılır).

Çalıştırma:
    python validation/compare_parametric.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from stoker_analytic import stoker  # noqa: F401  (idealize nokta için referans)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> None:
    print("[compare_parametric] İSKELET — Faz 3 doğrulama")
    print("-" * 56)
    print("Plan:")
    print("  1) Eğitilmiş parametrik modeli yükle (pinn_parametric).")
    print("  2) Test noktaları seç, ör.:")
    test_points = [
        {"B": 63.7, "t_f": 0.57, "h0": 26.9, "n": 0.04, "etiket": "Ürkmez"},
        {"B": 40.0, "t_f": 0.30, "h0": 15.0, "n": 0.03, "etiket": "örnek-2"},
    ]
    for p in test_points:
        print(f"       - {p['etiket']:>8}: "
              f"B={p['B']}, t_f={p['t_f']}, h0={p['h0']}, n={p['n']}")
    print("  3) pinn_parametric.predict(...) ile h(x), u(x) üret.")
    print("  4) İdealize/sürtünmesiz nokta için stoker() ile bağıl L2 hata hesapla;")
    print("     gerçekçi noktalar için referans araç çıktısıyla karşılaştır.")
    print("  5) Sonuçları results/ altına kaydet.")
    print("\nNot: Faz 1 (compare_baseline) doğrulandıktan ve pinn_parametric eğitimi")
    print("kurulduktan sonra doldurulacaktır.")

    # TODO (Faz 3): model yükleme + döngü + hata + grafik.
    _ = test_points  # iskelet


if __name__ == "__main__":
    main()
