"""Deneysel doğrulama — Vosoughi vd. (2021) ölçümleri vs analitik (Ritter/Stoker).

KATMAN 2 doğrulaması: gerçek laboratuvar dam-break ölçümleri. Bu script:
  1) temiz-su senaryosunu (data/processed/...csv) yükler,
  2) bilinmeyen kapı (gate) konumunu, analitik benzerlik çözümüne en küçük kareler
     ile OTURTARAK tahmin eder (early-time, t <= 1 s),
  3) ölçülen serbest su yüzeyi profillerini analitik çözümle üst üste çizer,
  4) RMSE raporlar.

Aynı (L, t) noktalarında eğitilmiş bir PINN tahmini de eklenebilir (TODO/hook):
PINN bu deneysel veriyi GÖRMEDEN çözer; sadece sonucunu üstüne koyup hatayı ölçeriz.

Çalıştırma:
    python validation/compare_experimental.py
(Önce: python src/data/parse_vosoughi2021.py ile CSV'leri üret.)
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stoker_analytic import analytic_profile  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
H_UP_CM = 30.0  # rezervuar başlangıç yüksekliği (Vosoughi 2021)
G = 9.81


def load_scenario(slug: str):
    """CSV'yi (L_cm sıralı, times sıralı, grid[nL, nT] cm) olarak döndürür."""
    path = os.path.join(PROC, f"vosoughi2021_{slug}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} yok. Önce: python src/data/parse_vosoughi2021.py")
    rows = list(csv.DictReader(open(path)))
    Ls = sorted({float(r["L_cm"]) for r in rows})
    ts = sorted({float(r["time_s"]) for r in rows})
    iL = {v: i for i, v in enumerate(Ls)}
    iT = {v: i for i, v in enumerate(ts)}
    grid = np.full((len(Ls), len(ts)), np.nan)
    h_down = float(rows[0]["h_down_cm"])
    for r in rows:
        grid[iL[float(r["L_cm"])], iT[float(r["time_s"])]] = float(r["level_cm"])
    return np.array(Ls), np.array(ts), grid, h_down


def fit_gate(Ls, ts, grid, h_down, t_max=1.0):
    """Analitik benzerlik çözümüne en küçük kareler ile kapı konumu (cm) tahmini."""
    x_m = Ls / 100.0
    h_up_m, h_down_m = H_UP_CM / 100.0, h_down / 100.0
    use_t = [j for j, t in enumerate(ts) if t <= t_max]

    best_x0, best_rmse = None, np.inf
    for x0_cm in np.arange(Ls.min(), Ls.max(), 2.0):
        se, cnt = 0.0, 0
        for j in use_t:
            h, _ = analytic_profile(x_m, ts[j], h_up_m, h_down_m, x0=x0_cm / 100.0, g=G)
            ana = h * 100.0  # cm
            meas = grid[:, j]
            m = ~np.isnan(meas)
            se += np.sum((ana[m] - meas[m]) ** 2)
            cnt += m.sum()
        rmse = np.sqrt(se / max(cnt, 1))
        if rmse < best_rmse:
            best_rmse, best_x0 = rmse, x0_cm
    return best_x0, best_rmse


def main() -> None:
    scenarios = [
        ("clearwater_dry", "Kuru mansap (Ritter)"),
        ("clearwater_wet2cm", "Islak mansap 2 cm (Stoker)"),
    ]
    plot_times = [0.4, 0.8, 2.0]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(13, 5))
    for ax, (slug, title) in zip(axes, scenarios):
        Ls, ts, grid, h_down = load_scenario(slug)
        x0_cm, rmse = fit_gate(Ls, ts, grid, h_down)
        print(f"[{slug}] kapı tahmini x0 ≈ {x0_cm:.0f} cm, "
              f"early-time RMSE = {rmse:.2f} cm  (h_down={h_down} cm)")

        x_fine = np.linspace(Ls.min(), Ls.max(), 300)
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(plot_times)))
        for c, t in zip(colors, plot_times):
            j = int(np.argmin(np.abs(ts - t)))
            ax.plot(Ls, grid[:, j], "o", color=c, ms=5,
                    label=f"deney t={ts[j]:.2f}s")
            h, _ = analytic_profile(x_fine / 100.0, ts[j], H_UP_CM / 100.0,
                                    h_down / 100.0, x0=x0_cm / 100.0, g=G)
            ax.plot(x_fine, h * 100.0, "-", color=c, lw=1.5, alpha=0.8)
        ax.axvline(x0_cm, ls=":", color="gray", lw=1, label=f"kapı x0≈{x0_cm:.0f}cm")
        ax.set(title=f"{title}\nRMSE={rmse:.1f} cm",
               xlabel="L (cm)", ylabel="su yüzeyi (cm)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Katman 2: Vosoughi vd. (2021) deneysel veri vs analitik "
                 "(○ deney, — analitik)")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "experimental_vs_analytic.png")
    fig.savefig(out, dpi=120)
    print(f"✓ Grafik kaydedildi: {out}")

    # TODO (PINN doğrulaması): bu senaryoyu PINN'e kur (h_up=0.30 m, h_down,
    # flume sürtünmesi n, alan uzunluğu ~5.52 m), model.predict ile aynı (L,t)
    # noktalarında tahmin üret ve deneysel ölçümle bağıl hatayı hesapla.


if __name__ == "__main__":
    main()
