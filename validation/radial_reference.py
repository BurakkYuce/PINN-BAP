"""Radyal (silindirik) dam-break — yarı-analitik kıyas referansı (2D sanity check).

2D'de tam analitik dam-break çözümü yoktur. Ancak eksenel simetrik (radyal) bir
dam-break, erken zamanda radyal yön boyunca yaklaşık olarak 1D düzlemsel
dam-break gibi davranır. Bu modül, mevcut 1D Stoker çözümünü (stoker_analytic.py)
radyal koordinat r = sqrt((x-xc)^2 + (y-yc)^2) boyunca eşleyerek bir referans verir.

UYARI (dürüstlük notu):
  * Düzlemsel Stoker, silindirik denklemdeki 1/r kaynak terimini İÇERMEZ.
  * Bu yüzden referans yalnızca ERKEN ZAMANDA ve şok cephesi CİVARINDA geçerlidir;
    merkeze yakın bölgede (drenaj) ve geç zamanda sapar.
  * Daha sağlam referans için yüksek-çözünürlüklü sonlu hacim (FV) çözümü
    eklenebilir (TODO).

Referans yaklaşımı: Tian vd. (2025, arXiv:2501.11372) radyal dam-break'i 2D SWE
PINN doğrulaması için kullanır.
"""
from __future__ import annotations

import numpy as np

from stoker_analytic import stoker  # aynı klasör (validation/)


def radial_profile(x, y, t: float, x_c: float, y_c: float, R0: float,
                   hl: float, hr: float, g: float = 9.81):
    """Verilen (x, y) noktaları ve t anı için radyal referansı döndürür.

    Döndürür
    --------
    r   : her noktanın merkeze uzaklığı
    h   : referans su derinliği (düzlemsel Stoker, kapı r=R0)
    u_r : referans radyal hız
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
    # 1D düzlemsel Stoker: "kapı" r = R0; iç (r<R0)=hl, dış (r>R0)=hr.
    h, u_r = stoker(r, t, hl=hl, hr=hr, x0=R0, g=g)
    return r, h, u_r


if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_c, y_c, R0, hl, hr, g = 5.0, 5.0, 1.5, 1.0, 0.5, 9.81
    # Merkezden dışa radyal hat (y = y_c)
    r_line = np.linspace(0.0, 5.0, 300)
    xs = x_c + r_line
    ys = np.full_like(xs, y_c)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in (0.2, 0.4, 0.6):
        _, h, _ = radial_profile(xs, ys, t, x_c, y_c, R0, hl, hr, g)
        ax.plot(r_line, h, lw=2, label=f"t = {t} s")
    ax.axvline(R0, ls=":", color="gray", label=f"R0={R0} m")
    ax.set(title="Radyal dam-break referansı (düzlemsel Stoker eşlemesi)",
           xlabel="r (m)", ylabel="h (m)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "2d", "radial_reference.png")
    fig.savefig(os.path.normpath(out), dpi=120)
    print(f"✓ Radyal referans profili kaydedildi: {os.path.normpath(out)}")
    print("  (Uyarı: yalnızca erken-zaman / şok-cephesi civarı geçerli; 1/r etkisi yok.)")
