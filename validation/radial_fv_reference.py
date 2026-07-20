"""Radyal (silindirik) dam-break — 1D sonlu-hacim (FV) referans çözücüsü.

Düzlemsel Stoker (radial_reference.py) silindirik denklemdeki 1/r kaynak terimini
İÇERMEZ; bu yüzden yalnızca erken-zaman/cephe civarı geçerlidir. Bu modül, eksenel
simetrik sığ su denklemlerini geometrik kaynak terimiyle birlikte FV ile çözer ve
MERKEZ DAHİL her yerde geçerli bir "gerçek 2D" referans verir:

    h_t + (h u)_r          = -(h u)/r
    (h u)_t + (h u^2 + g h^2/2)_r = -(h u^2)/r

Şema: Rusanov (yerel Lax-Friedrichs) akı + SSP-RK2 zaman adımı, r=0'da yansıtıcı
(simetri) duvarı, dış sınırda transmissive. Islak yatak (hr>0) olduğundan kuru-yatak
işlemine gerek yoktur.

Bu, 2D PINN için temiz/yayınlanabilir bir L2 referansı sağlar.
"""
from __future__ import annotations

import numpy as np


def solve_radial_fv(t_eval, hl: float = 1.0, hr: float = 0.5, R0: float = 1.5,
                    g: float = 9.81, R_max: float = 5.0, N: int = 2000,
                    cfl: float = 0.4):
    """Eksenel simetrik SWE'yi FV ile çözer; istenen anlarda (h,u) anlık görüntüsü.

    Döndürür
    --------
    r      : hücre merkezleri (N,)
    snaps  : {t: (h, u)} sözlüğü (istenen her t için)
    """
    t_eval = sorted(float(t) for t in t_eval)
    dr = R_max / N
    r = (np.arange(N) + 0.5) * dr
    h = np.where(r < R0, hl, hr).astype(float)
    hu = np.zeros(N)

    def rhs(h, hu):
        u = hu / h
        # iç ara-yüz Rusanov akıları (hücre i ile i+1 arası), i=0..N-2
        hL, huL, uL = h[:-1], hu[:-1], u[:-1]
        hR, huR, uR = h[1:], hu[1:], u[1:]
        Fm_L, Fmom_L = huL, huL * uL + 0.5 * g * hL * hL
        Fm_R, Fmom_R = huR, huR * uR + 0.5 * g * hR * hR
        a = np.maximum(np.abs(uL) + np.sqrt(g * hL), np.abs(uR) + np.sqrt(g * hR))
        Fm_int = 0.5 * (Fm_L + Fm_R) - 0.5 * a * (hR - hL)
        Fmom_int = 0.5 * (Fmom_L + Fmom_R) - 0.5 * a * (huR - huL)

        Fm = np.empty(N + 1)
        Fmom = np.empty(N + 1)
        Fm[1:N], Fmom[1:N] = Fm_int, Fmom_int
        # r=0 yansıtıcı duvar: kütle akısı 0, momentum akısı yalnız basınç
        Fm[0], Fmom[0] = 0.0, 0.5 * g * h[0] ** 2
        # dış sınır transmissive (sıfır-gradyan)
        Fm[N], Fmom[N] = hu[-1], hu[-1] * u[-1] + 0.5 * g * h[-1] ** 2

        dh = -(Fm[1:] - Fm[:-1]) / dr - hu / r
        dhu = -(Fmom[1:] - Fmom[:-1]) / dr - hu * u / r
        return dh, dhu

    snaps = {}
    targets = list(t_eval)
    t = 0.0
    while targets:
        smax = np.max(np.abs(hu / h) + np.sqrt(g * h))
        dt = cfl * dr / smax
        if t + dt >= targets[0]:
            dt = targets[0] - t
        # SSP-RK2 (Heun)
        k1h, k1hu = rhs(h, hu)
        h1, hu1 = h + dt * k1h, hu + dt * k1hu
        k2h, k2hu = rhs(h1, hu1)
        h = h + 0.5 * dt * (k1h + k2h)
        hu = hu + 0.5 * dt * (k1hu + k2hu)
        t += dt
        if abs(t - targets[0]) < 1e-12:
            snaps[targets.pop(0)] = (h.copy(), (hu / h).copy())
    return r, snaps


def reference_profiles(r_query, t_eval, hl=1.0, hr=0.5, R0=1.5, g=9.81,
                       R_max=5.0, N=2000, cfl=0.4):
    """İstenen radyal konumlarda (r_query) ve anlarda FV referansını döndürür.

    Döndürür: {t: (h, u)} — her dizi r_query üzerinde enterpole edilmiş.
    """
    r_fv, snaps = solve_radial_fv(t_eval, hl, hr, R0, g, R_max, N, cfl)
    out = {}
    for t in t_eval:
        h, u = snaps[t]
        out[float(t)] = (np.interp(r_query, r_fv, h), np.interp(r_query, r_fv, u))
    return out


if __name__ == "__main__":
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys_path = os.path.dirname(__file__)
    import sys
    sys.path.insert(0, sys_path)
    from stoker_analytic import stoker  # düzlemsel kıyas

    hl, hr, R0, g = 1.0, 0.5, 1.5, 9.81
    times = [0.2, 0.4, 0.6]
    r, snaps = solve_radial_fv(times, hl, hr, R0, g, R_max=5.0, N=2000)

    assert all(np.isfinite(snaps[t][0]).all() for t in times), "FV çözümü NaN üretti!"
    print("Radyal FV referans çözücü — self-test")
    print(f"  hl={hl}, hr={hr}, R0={R0}, N=2000")
    for t in times:
        hmin, hmax = snaps[t][0].min(), snaps[t][0].max()
        print(f"  t={t}s: h aralığı [{hmin:.3f}, {hmax:.3f}] m (merkez h={snaps[t][0][0]:.3f})")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(times)))
    for c, t in zip(colors, times):
        ax.plot(r, snaps[t][0], "-", color=c, lw=2, label=f"FV (2D) t={t}s")
        h_pl, _ = stoker(r, t, hl=hl, hr=hr, x0=R0, g=g)
        ax.plot(r, h_pl, "--", color=c, lw=1, alpha=0.6,
                label=f"düzlemsel Stoker t={t}s")
    ax.axvline(R0, ls=":", color="gray")
    ax.set(title="Radyal FV (—) vs düzlemsel Stoker (--): merkezde 1/r farkı",
           xlabel="r (m)", ylabel="h (m)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "results", "2d", "radial_fv_reference.png")
    fig.savefig(os.path.normpath(out), dpi=120)
    print(f"  ✓ FV vs düzlemsel grafiği: {os.path.normpath(out)}")
    print("  (FV merkezi boşaltır, düzlemsel boşaltmaz -> asıl fark budur.)")
