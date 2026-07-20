"""Faz 2A doğrulama — 2D PINN vs radyal FV referansı (azimutal ortalamalı).

`pinn_baseline_2d.py` çıktısını (results/2d/baseline_2d_prediction.npz) okur ve:
  (a) sabit t'lerde h(x,y) heatmap'i çizer,
  (b) merkezden dışa radyal profili **azimutal ortalama** ile çıkarır (tek ışın
      yerine N_theta açıda örnekleyip ortalar) ve GERÇEK 2D referansla
      (radial_fv_reference, 1/r kaynak terimli FV) karşılaştırır,
  (c) radyal yönde bağıl L2 hatasını HEM tek-ışın HEM azimutal-ortalama için
      raporlar; ayrıca **açısal asimetri** ("kelebek") metriğini yazdırır.

Neden azimutal ortalama: tek ışın (yalnız +x) açısal yanlılık taşır. Tüm açıları
ortalamak temiz/adil bir radyal profil verir; açılar arası std ise asimetriyi
(kelebeği) NİCELLEŞTİRİR — kozmetik değil, ölçülebilir bir sayı.

Çalıştırma:
    python validation/compare_baseline_2d.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from radial_fv_reference import reference_profiles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def azimuthal_profile(h_grid, x, y, x_c, y_c, r_query, n_theta=72):
    """h(x,y) ızgarasından, merkez (x_c,y_c) etrafında azimutal profil.

    Her r için n_theta açıda bilineer enterpolasyonla h örneklenir.
    Döndürür: (mean, std) — açılar üzerinden ortalama ve standart sapma (asimetri).
    """
    interp = RegularGridInterpolator((y, x), h_grid, bounds_error=False,
                                     fill_value=None)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    mean = np.empty_like(r_query, dtype=float)
    std = np.empty_like(r_query, dtype=float)
    for i, r in enumerate(r_query):
        xs = x_c + r * np.cos(thetas)
        ys = y_c + r * np.sin(thetas)
        vals = interp(np.column_stack([ys, xs]))  # RGI sırası: (y, x)
        mean[i] = np.nanmean(vals)
        std[i] = np.nanstd(vals)
    return mean, std


def main() -> None:
    npz = os.path.join(ROOT, "results", "2d", "baseline_2d_prediction.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"{npz} yok. Önce: python src/pinn/pinn_baseline_2d.py [--smoke]")

    P = np.load(npz)
    x, y, times = P["x"], P["y"], np.atleast_1d(P["t"])
    H = P["h"]
    x_c, y_c, R0 = float(P["x_c"]), float(P["y_c"]), float(P["R0"])
    hl, hr, g = float(P["hl"]), float(P["hr"]), float(P["g"])

    # (a) Heatmap'ler
    fig, axes = plt.subplots(1, len(times), figsize=(5 * len(times), 4.3), squeeze=False)
    for ax, k in zip(axes[0], range(len(times))):
        im = ax.pcolormesh(x, y, H[k], shading="auto", cmap="viridis", vmin=hr, vmax=hl)
        ax.set(title=f"h(x,y), t={times[k]:.2f}s", xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="h (m)")
    fig.suptitle("2D PINN — dışa yayılan radyal dam-break dalgası")
    fig.tight_layout()
    out_hm = os.path.join(ROOT, "results", "2d", "h_heatmap.png")
    fig.savefig(out_hm, dpi=120)

    # (b)+(c) Radyal profil: azimutal ortalama (yoğun r) + tek ışın kıyası
    # Tüm açılarda alan içinde kalmak için r_max = merkeze en yakın kenara mesafe.
    r_max = min(x_c - x.min(), x.max() - x_c, y_c - y.min(), y.max() - y_c)
    r_max = 0.9 * r_max                       # kenar BC bölgesinden uzak dur
    r_dense = np.linspace(0.0, r_max, 400)     # yoğun r — ızgara çözünürlüğüne bağlı değil

    # tek ışın (+x, y=y_c satırı) — eski yöntemin kıyası için
    iyc = int(np.argmin(np.abs(y - y_c)))
    ixc = int(np.argmin(np.abs(x - x_c)))
    r_ray = x[ixc:] - x_c

    fv_dense = reference_profiles(r_dense, list(times), hl=hl, hr=hr, R0=R0, g=g,
                                  R_max=float(r_dense.max()), N=2000)
    fv_ray = reference_profiles(r_ray, list(times), hl=hl, hr=hr, R0=R0, g=g,
                                R_max=float(r_ray.max()), N=2000)

    print("Faz 2A — 2D PINN vs radyal FV (azimutal ortalama, gerçek 2D 1/r dahil)")
    print("-" * 68)
    print(f"  azimutal örnekleme: 72 açı, yoğun r (400 nokta), r_max={r_max:.2f} m")
    print(f"  {'t':>5} | {'tek-ışın L2':>12} | {'azimutal L2':>12} | "
          f"{'maks açısal asimetri':>22}")
    print("  " + "-" * 64)

    fig2, ax2 = plt.subplots(figsize=(8.5, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(times)))
    for c, k in zip(colors, range(len(times))):
        # azimutal ortalama profil + asimetri bandı
        mean, std = azimuthal_profile(H[k], x, y, x_c, y_c, r_dense, n_theta=72)
        h_fv_d = fv_dense[float(times[k])][0]
        relL2_azi = np.linalg.norm(mean - h_fv_d) / (np.linalg.norm(h_fv_d) + 1e-12)

        # tek ışın
        h_ray = H[k, iyc, ixc:]
        h_fv_r = fv_ray[float(times[k])][0]
        relL2_ray = np.linalg.norm(h_ray - h_fv_r) / (np.linalg.norm(h_fv_r) + 1e-12)

        # asimetri ("kelebek"): açılar arası maks std, (hl-hr)'ye oranla %
        asym_m = float(std.max())
        asym_pct = 100.0 * asym_m / (hl - hr)
        r_at = r_dense[int(np.argmax(std))]
        print(f"  {times[k]:5.2f} | {relL2_ray * 100:10.1f} % | "
              f"{relL2_azi * 100:10.1f} % | "
              f"{asym_m:6.3f} m = {asym_pct:4.1f}% (r={r_at:.2f}m)")

        ax2.plot(r_dense, h_fv_d, "-", color=c, lw=1.6, alpha=0.65,
                 label=f"FV t={times[k]:.2f}s")
        ax2.plot(r_dense, mean, "--", color=c, lw=2,
                 label=f"PINN (azimutal) t={times[k]:.2f}s")
        ax2.fill_between(r_dense, mean - std, mean + std, color=c, alpha=0.15)

    ax2.axvline(R0, ls=":", color="gray", lw=1, label=f"R0={R0}m")
    ax2.set(title="Radyal profil: PINN azimutal ortalama (--, ±std bandı) vs 2D FV (—)",
            xlabel="r (m)", ylabel="h (m)")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out_rc = os.path.join(ROOT, "results", "2d", "radial_cut.png")
    fig2.savefig(out_rc, dpi=120)

    print("  " + "-" * 64)
    print("  Yorum: azimutal L2 ~ tek-ışın L2 ve asimetri küçükse 'kelebek' yok —")
    print("         kalan fark çözüm (şok yayması/plato), örnekleme değil. 1D radyal")
    print("         PINN (pinn_radial_1d.py) bunu denklem düzeyinde keskinleştirir.")
    print(f"  ✓ Heatmap     : {out_hm}")
    print(f"  ✓ Radyal kesit: {out_rc}")


if __name__ == "__main__":
    main()
