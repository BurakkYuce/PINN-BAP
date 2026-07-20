"""PINN vs GERÇEK ölçüm (Vosoughi 2021) — sürtünmeli flume dam-break doğrulaması.

Bu, projenin asıl kanıtı: PINN deneysel veriyi GÖRMEDEN (yalnızca denklem + IC/BC)
çözer; sonra aynı 20 ölçüm konumu × zaman noktalarında laboratuvar ölçümleriyle
karşılaştırılır. Manning sürtünmesi sayesinde PINN, sürtünmesiz analitikten daha
gerçekçi olmalıdır (özellikle geç zamanlarda).

Ön koşul:
    python src/data/parse_vosoughi2021.py                                  # ölçüm CSV
    DDE_BACKEND=pytorch python src/pinn/pinn_baseline.py --scenario experimental
Çalıştırma:
    python validation/compare_experimental_pinn.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stoker_analytic import analytic_profile  # noqa: E402
from compare_experimental import load_scenario, fit_gate, H_UP_CM, G  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIO_SLUG = "clearwater_wet2cm"   # PINN config'i ile eşleşen senaryo (ıslak 2 cm)


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def main() -> None:
    # --- ölçüm + PINN tahmini yükle ---
    Ls, ts_meas, grid, h_down = load_scenario(SCENARIO_SLUG)  # cm, s, cm
    x0_cm, _ = fit_gate(Ls, ts_meas, grid, h_down)            # analitik kapı tahmini

    npz = os.path.join(ROOT, "results", "experimental_prediction.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"{npz} yok. Önce: python src/pinn/pinn_baseline.py --scenario experimental")
    P = np.load(npz)
    px = P["x"]              # m (flume başı orijinli, ölçüm L ile aynı eksen)
    pt = np.atleast_1d(P["t"])
    pH = P["h"]              # m, şekil [nt, nx]

    gauge_x_m = Ls / 100.0

    # PINN ve analitiği, ölçümle ortak zamanlarda kıyasla
    common_t = [t for t in pt if np.any(np.isclose(ts_meas, t, atol=1e-6))]
    pinn_all, ana_all, meas_all = [], [], []
    for t in common_t:
        kp = int(np.argmin(np.abs(pt - t)))
        jm = int(np.argmin(np.abs(ts_meas - t)))
        pinn_cm = np.interp(gauge_x_m, px, pH[kp]) * 100.0
        ana_h, _ = analytic_profile(gauge_x_m, t, H_UP_CM / 100.0, h_down / 100.0,
                                    x0=x0_cm / 100.0, g=G)
        ana_cm = ana_h * 100.0
        meas_cm = grid[:, jm]
        pinn_all.append(pinn_cm); ana_all.append(ana_cm); meas_all.append(meas_cm)

    pinn_all = np.concatenate(pinn_all)
    ana_all = np.concatenate(ana_all)
    meas_all = np.concatenate(meas_all)

    rmse_pinn = rmse(pinn_all, meas_all)
    rmse_ana = rmse(ana_all, meas_all)

    print(f"PINN vs ölçüm (Vosoughi 2021, {SCENARIO_SLUG})")
    print("-" * 52)
    print(f"  Karşılaştırılan zamanlar: {[f'{t:.2f}' for t in common_t]} s")
    print(f"  Nokta sayısı            : {(~np.isnan(meas_all)).sum()} (20 konum × {len(common_t)} an)")
    print(f"  RMSE  PINN  vs ölçüm    : {rmse_pinn:5.2f} cm")
    print(f"  RMSE  analitik vs ölçüm : {rmse_ana:5.2f} cm  (sürtünmesiz Stoker)")

    # --- grafik ---
    plot_times = [t for t in (0.2, 0.4, 0.8) if t in common_t] or common_t[:3]
    x_fine = np.linspace(Ls.min(), Ls.max(), 300)
    fig, axes = plt.subplots(1, len(plot_times), figsize=(5 * len(plot_times), 4.5),
                             squeeze=False)
    for ax, t in zip(axes[0], plot_times):
        kp = int(np.argmin(np.abs(pt - t)))
        jm = int(np.argmin(np.abs(ts_meas - t)))
        ax.plot(Ls, grid[:, jm], "ko", ms=5, label="ölçüm")
        ax.plot(px * 100.0, pH[kp] * 100.0, "r-", lw=2, label="PINN")
        ana_h, _ = analytic_profile(x_fine / 100.0, t, H_UP_CM / 100.0, h_down / 100.0,
                                    x0=x0_cm / 100.0, g=G)
        ax.plot(x_fine, ana_h * 100.0, "b--", lw=1.5, alpha=0.7, label="Stoker (analitik)")
        ax.set(title=f"t = {t:.2f} s", xlabel="L (cm)", ylabel="su yüzeyi (cm)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"PINN (sürtünmeli) vs ölçüm vs analitik — "
                 f"RMSE: PINN={rmse_pinn:.1f} cm, analitik={rmse_ana:.1f} cm")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "experimental_pinn_vs_measured.png")
    fig.savefig(out, dpi=120)
    print(f"  ✓ Grafik kaydedildi: {out}")


if __name__ == "__main__":
    main()
