"""PINN çıktısından doğal dil afet raporu — İSKELET.

PINN saniyeler içinde h(x,t), u(x,t) alanını üretir; bu modül bu alandan afet
yönetimi için anlamlı metrikleri (maks su derinliği, taşkın varış süresi, pik hız,
sular altında kalan mesafe) çıkarır ve Türkçe bir özet rapor üretir.

Şu an şablon tabanlıdır (harici bağımlılık yok). `use_llm=True` için bir LLM
(ör. Anthropic Claude) çağrısı eklenecek hook bırakılmıştır.
"""
from __future__ import annotations

import numpy as np


def extract_metrics(x, h, u, t: float, depth_threshold: float = 0.1) -> dict:
    """h(x), u(x) profilinden afet metriklerini çıkarır.

    depth_threshold : taşkın sayılan minimum su derinliği (m).
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    u = np.asarray(u, dtype=float)

    flooded = h > depth_threshold
    inundation_extent = float(x[flooded].max() - x[flooded].min()) if flooded.any() else 0.0

    return {
        "time_s": float(t),
        "max_depth_m": float(h.max()),
        "max_depth_location_m": float(x[int(np.argmax(h))]),
        "peak_velocity_ms": float(np.abs(u).max()),
        "inundation_extent_m": inundation_extent,
        "depth_threshold_m": depth_threshold,
    }


def generate_report(metrics: dict, use_llm: bool = False) -> str:
    """Metrik sözlüğünden Türkçe afet raporu üretir."""
    if use_llm:
        # TODO (LLM hook): Anthropic Claude API ile metrikleri zengin, bağlamsal
        # bir afet raporuna dönüştür. Şablon çıktısını few-shot örnek olarak ver.
        raise NotImplementedError(
            "LLM tabanlı rapor henüz bağlanmadı — şablon için use_llm=False kullanın.")

    return (
        "TAŞKIN DURUM RAPORU (PINN tahmini)\n"
        "===================================\n"
        f"Değerlendirme anı           : t = {metrics['time_s']:.0f} s\n"
        f"Maksimum su derinliği       : {metrics['max_depth_m']:.2f} m "
        f"(x ≈ {metrics['max_depth_location_m']:.1f} m)\n"
        f"Pik akış hızı               : {metrics['peak_velocity_ms']:.2f} m/s\n"
        f"Sular altında kalan mesafe  : {metrics['inundation_extent_m']:.1f} m "
        f"(eşik {metrics['depth_threshold_m']:.2f} m)\n"
        "-----------------------------------\n"
        "Not: Bu rapor PINN tahmininden otomatik üretilmiştir; saha doğrulaması "
        "ve idari değerlendirme gerektirir."
    )


if __name__ == "__main__":
    # Örnek (yapay) metriklerle iskelet gösterimi
    x = np.linspace(-10, 10, 200)
    h = 0.5 + 0.5 * (x < 2)
    u = np.where(x < 2, 1.5, 0.0)
    m = extract_metrics(x, h, u, t=60.0)
    print(generate_report(m, use_llm=False))
