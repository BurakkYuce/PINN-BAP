"""Stoker (1957) ıslak-yatak dam-break analitik çözümü — matematiksel ground truth.

hl > hr > 0 koşulunda 1D sığ su denklemlerinin tam Riemann çözümü. Yapı:

    sol sabit (hl) | rarefaction yelpazesi | orta sabit (hm, um) | shock | sağ sabit (hr)

Orta sabit durum (hm, um), rarefaction (sol) ve shock (sağ) hız eşlemelerinin
eşitlenmesiyle bulunur:
    rarefaction:  um = 2*( sqrt(g*hl) - sqrt(g*hm) )
    shock (R-H):  um = (hm - hr) * sqrt( g*(hm + hr) / (2*hm*hr) )

PINN doğrulamasında (Faz 1) referans çözüm olarak kullanılır.

Referans: Stoker, J.J. (1957). "Water Waves: The Mathematical Theory with
Applications." Interscience.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _middle_state(hl: float, hr: float, g: float) -> tuple[float, float]:
    """Orta sabit durumun derinliği hm ve hızı um."""
    cl = np.sqrt(g * hl)

    def residual(hm: float) -> float:
        u_raref = 2.0 * (cl - np.sqrt(g * hm))
        u_shock = (hm - hr) * np.sqrt(g * (hm + hr) / (2.0 * hm * hr))
        return u_raref - u_shock

    # hm, hr ile hl arasında yer alır.
    hm = brentq(residual, hr * (1.0 + 1e-12), hl * (1.0 - 1e-12))
    um = 2.0 * (cl - np.sqrt(g * hm))
    return hm, um


def stoker(x, t: float, hl: float = 1.0, hr: float = 0.5,
           x0: float = 0.0, g: float = 9.81):
    """Verilen x dizisi ve t anı için (h, u) profillerini döndürür.

    x  : konum dizisi (m)
    t  : zaman (s)
    hl : sol (rezervuar) derinliği; hr : sağ (mansap) derinliği (hl > hr > 0)
    x0 : membran/baraj konumu
    g  : yerçekimi ivmesi
    """
    x = np.asarray(x, dtype=float)
    h = np.empty_like(x)
    u = np.empty_like(x)

    if t <= 0.0:
        h[:] = np.where(x < x0, hl, hr)
        u[:] = 0.0
        return h, u

    hm, um = _middle_state(hl, hr, g)
    cl = np.sqrt(g * hl)
    cm = np.sqrt(g * hm)
    shock_speed = um * hm / (hm - hr)  # Rankine-Hugoniot

    xi = (x - x0) / t  # benzerlik değişkeni x/t

    for k, s in enumerate(xi):
        if s <= -cl:
            # 1) sol sabit durum
            h[k], u[k] = hl, 0.0
        elif s <= um - cm:
            # 2) rarefaction yelpazesi
            c = (2.0 * cl - s) / 3.0
            h[k] = c * c / g
            u[k] = 2.0 / 3.0 * (cl + s)
        elif s <= shock_speed:
            # 3) orta sabit durum
            h[k], u[k] = hm, um
        else:
            # 4) sağ sabit durum (shock ardı)
            h[k], u[k] = hr, 0.0

    return h, u


def ritter(x, t: float, h0: float = 1.0, x0: float = 0.0, g: float = 9.81):
    """Ritter (1892) KURU-yatak dam-break analitik çözümü.

    Rezervuar solda (x < x0, derinlik h0), mansap kuru (h = 0). Sürtünmesiz.
    Stoker'ın hr -> 0 limiti; ıslak-yatak çözümü kuru yatakta tekildir, bu yüzden
    kuru senaryolar için Ritter kullanılır.

        s = (x - x0)/t
        s <= -c0          : h = h0, u = 0           (rezervuar)
        -c0 < s < 2*c0    : h = (2*c0 - s)^2/(9g),  u = 2*(s + c0)/3   (rarefaction)
        s >= 2*c0         : h = 0, u = 0            (kuru)
    """
    x = np.asarray(x, dtype=float)
    h = np.empty_like(x)
    u = np.empty_like(x)

    if t <= 0.0:
        h[:] = np.where(x < x0, h0, 0.0)
        u[:] = 0.0
        return h, u

    c0 = np.sqrt(g * h0)
    s = (x - x0) / t
    for k, sk in enumerate(s):
        if sk <= -c0:
            h[k], u[k] = h0, 0.0
        elif sk < 2.0 * c0:
            h[k] = (2.0 * c0 - sk) ** 2 / (9.0 * g)
            u[k] = 2.0 * (sk + c0) / 3.0
        else:
            h[k], u[k] = 0.0, 0.0
    return h, u


def analytic_profile(x, t, h_up, h_down, x0=0.0, g=9.81):
    """Mansap derinliğine göre uygun analitik çözümü seçer.

    h_down > 0 -> Stoker (ıslak yatak); h_down == 0 -> Ritter (kuru yatak).
    """
    if h_down > 0.0:
        return stoker(x, t, hl=h_up, hr=h_down, x0=x0, g=g)
    return ritter(x, t, h0=h_up, x0=x0, g=g)


if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hl, hr, x0, g, t = 1.0, 0.5, 0.0, 9.81, 1.0
    x = np.linspace(-10.0, 10.0, 400)
    h, u = stoker(x, t, hl=hl, hr=hr, x0=x0, g=g)
    hm, um = _middle_state(hl, hr, g)

    print("Stoker ıslak-yatak dam-break — self-test")
    print(f"  hl={hl}, hr={hr}, t={t}s")
    print(f"  orta durum: hm={hm:.4f} m, um={um:.4f} m/s, shock_speed={um*hm/(hm-hr):.4f} m/s")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x, h, lw=2)
    ax[0].set(title=f"Su derinliği h (t={t}s)", xlabel="x (m)", ylabel="h (m)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(x, u, lw=2, color="tab:red")
    ax[1].set(title=f"Hız u (t={t}s)", xlabel="x (m)", ylabel="u (m/s)")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Stoker ıslak-yatak dam-break analitik çözümü")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "..", "results", "stoker_analytic.png")
    fig.savefig(os.path.normpath(out), dpi=120)
    print(f"  ✓ Profil kaydedildi: {os.path.normpath(out)}")
