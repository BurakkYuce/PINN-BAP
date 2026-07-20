"""Froehlich (2008) baraj kırılma (breach) parametre denklemleri.

⚠ KAPSAM: Bu denklemler YALNIZCA toprak/kaya dolgu (embankment) barajlar içindir —
kademeli (aşamalı) kırılma rejimi. BETON (gravite/kemer) barajlara UYGULANMAZ;
onlar ani / temel-yapısal yıkılır (t_f ≈ 0). Bkz. docs/baraj_tipleri.md.

Overtopping kaynaklı kırılma için ortalama kırılma genişliği (B_ave) ve kırılma
oluşum süresini (t_f) ampirik olarak hesaplar. Bu iki büyüklük:
  * parametrik PINN'in girdi uzayını (B, t_f) fiziksel olarak konumlandırır,
  * gerçek (dolgu) baraj vakalarını (ör. Ürkmez, Korkuteli) parametre uzayında
    bir nokta yapar.

Denklemler:
    B_ave = 0.27 * K0 * Vw^0.32 * hb^0.04        (m)
    t_f   = 0.0176 * sqrt(Vw / (g * hb^2))        (saat)
    K0    = 1.3  (overtopping)  |  1.0  (piping vb.)

Referans:
    Froehlich, D.C. (2008). "Embankment Dam Breach Parameters and Their
    Uncertainties." J. Hydraulic Engineering, 134(12), 1708-1721.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

G = 9.81  # yerçekimi ivmesi (m/s^2)


@dataclass
class BreachParams:
    """Froehlich kırılma parametreleri sonucu."""

    B_ave: float        # ortalama kırılma genişliği (m)
    t_f_hours: float    # kırılma oluşum süresi (saat)
    t_f_seconds: float  # kırılma oluşum süresi (s)
    K0: float           # overtopping katsayısı
    Vw: float           # rezervuar su hacmi (m^3)
    hb: float           # kırılma yüksekliği (m)

    def as_dict(self) -> dict:
        return asdict(self)


# Froehlich'in geçerli olduğu (kademeli kırılan) dolgu baraj tipleri
EMBANKMENT_TYPES = {"earth-fill", "rockfill", "embankment", "toprak", "kaya-dolgu"}


def breach_params(Vw: float, hb: float, overtopping: bool = True,
                  dam_type: str | None = None) -> BreachParams:
    """Froehlich (2008) kırılma parametrelerini hesaplar.

    Parametreler
    ------------
    Vw : float
        Kırılma anında rezervuar su hacmi (m^3).
    hb : float
        Kırılma yüksekliği (m).
    overtopping : bool
        True ise K0 = 1.3 (overtopping); aksi halde K0 = 1.0.
    dam_type : str | None
        Verilirse tip kontrol edilir; beton barajlar için hata verir (Froehlich
        yalnız dolgu barajlar içindir). None ise kontrol atlanır.
    """
    if Vw <= 0.0 or hb <= 0.0:
        raise ValueError("Vw ve hb pozitif olmalı.")
    if dam_type is not None and dam_type.lower() not in EMBANKMENT_TYPES:
        raise ValueError(
            f"Froehlich yalnızca dolgu barajlar içindir; '{dam_type}' uygun değil. "
            "Beton barajlar ani dam-break ile ele alınır (bkz. docs/baraj_tipleri.md).")

    K0 = 1.3 if overtopping else 1.0
    B_ave = 0.27 * K0 * (Vw ** 0.32) * (hb ** 0.04)
    t_f_hours = 0.0176 * math.sqrt(Vw / (G * hb ** 2))

    return BreachParams(
        B_ave=B_ave,
        t_f_hours=t_f_hours,
        t_f_seconds=t_f_hours * 3600.0,
        K0=K0,
        Vw=Vw,
        hb=hb,
    )


if __name__ == "__main__":
    # Ürkmez Barajı testi (Haltas, Tayfur & Elci 2016 — Tablo 1)
    urkmez = breach_params(Vw=7.6e6, hb=26.9, overtopping=True)

    print("Ürkmez Barajı — Froehlich (2008) kırılma parametreleri")
    print("-" * 52)
    print(f"  Vw   = {urkmez.Vw:.3e} m^3")
    print(f"  hb   = {urkmez.hb:.1f} m")
    print(f"  K0   = {urkmez.K0}")
    print(f"  B_ave= {urkmez.B_ave:6.1f} m      (beklenen ~63.7 m)")
    print(f"  t_f  = {urkmez.t_f_hours:6.2f} saat  ({urkmez.t_f_seconds:.0f} s)  (beklenen ~0.57 saat)")

    assert abs(urkmez.B_ave - 63.7) < 1.0, "B_ave Ürkmez referansından sapıyor!"
    assert abs(urkmez.t_f_hours - 0.57) < 0.05, "t_f Ürkmez referansından sapıyor!"
    print("-" * 52)
    print("  ✓ Ürkmez referans değerleri doğrulandı.")
