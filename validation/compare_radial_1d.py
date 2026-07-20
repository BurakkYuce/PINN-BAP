"""Faz 2A (radyal) doğrulama — 1D radyal (r,t) PINN vs radyal FV referansı.

`pinn_radial_1d.py` çıktısını (results/2d/radial_1d_prediction.npz) okur ve aynı
silindirik formülasyondaki FV referansıyla (radial_fv_reference) karşılaştırır.
Aynı denklem ailesi → EN TEMİZ L2; ayrıca varsa 2D Kartezyen azimutal sonucuyla
yan yana kıyas yazdırır.

Çalıştırma:
    python validation/compare_radial_1d.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from radial_fv_reference import reference_profiles  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cartesian_azimuthal_L2(times, r_query, hl, hr, R0, g):
    """Varsa 2D Kartezyen baseline'ın azimutal-ortalama L2'sini döndürür (kıyas).

    {t: relL2} sözlüğü; npz yoksa None.
    """
    npz = os.path.join(ROOT, "results", "2d", "baseline_2d_prediction.npz")
    if not os.path.exists(npz):
        return None
    P = np.load(npz)
    x, y, H = P["x"], P["y"], P["h"]
    x_c, y_c = float(P["x_c"]), float(P["y_c"])
    t2d = np.atleast_1d(P["t"])
    fv = reference_profiles(r_query, list(times), hl=hl, hr=hr, R0=R0, g=g,
                            R_max=float(r_query.max()), N=2000)
    thetas = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    out = {}
    for t in times:
        idx = int(np.argmin(np.abs(t2d - t)))
        interp = RegularGridInterpolator((y, x), H[idx], bounds_error=False,
                                         fill_value=None)
        mean = np.array([
            np.nanmean(interp(np.column_stack([y_c + r * np.sin(thetas),
                                               x_c + r * np.cos(thetas)])))
            for r in r_query])
        h_fv = fv[float(t)][0]
        out[float(t)] = np.linalg.norm(mean - h_fv) / (np.linalg.norm(h_fv) + 1e-12)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="1D radyal PINN vs FV kıyas")
    parser.add_argument("--tag", type=str, default="",
                        help="radial_1d<tag>_prediction.npz oku (ör. '_lownu')")
    args = parser.parse_args()

    npz = os.path.join(ROOT, "results", "2d", f"radial_1d{args.tag}_prediction.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"{npz} yok. Önce: DDE_BACKEND=pytorch python src/pinn/pinn_radial_1d.py [--smoke] [--tag {args.tag}]")

    P = np.load(npz)
    r, times, H = P["r"], np.atleast_1d(P["t"]), P["h"]
    R0, hl, hr, g = float(P["R0"]), float(P["hl"]), float(P["hr"]), float(P["g"])

    fv = reference_profiles(r, list(times), hl=hl, hr=hr, R0=R0, g=g,
                            R_max=float(r.max()), N=2000)
    cart = _cartesian_azimuthal_L2(times, r, hl, hr, R0, g)

    print("Faz 2A (radyal) — 1D radyal PINN vs radyal FV (aynı formülasyon)")
    print("-" * 64)
    hdr = f"  {'t':>5} | {'1D-radyal PINN L2':>18}"
    if cart is not None:
        hdr += f" | {'2D Kartezyen (azimutal) L2':>26}"
    print(hdr)
    print("  " + "-" * (60 if cart is not None else 28))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(times)))
    for c, k in zip(colors, range(len(times))):
        h_pinn = H[k]
        h_fv = fv[float(times[k])][0]
        relL2 = np.linalg.norm(h_pinn - h_fv) / (np.linalg.norm(h_fv) + 1e-12)
        line = f"  {times[k]:5.2f} | {relL2 * 100:16.1f} %"
        if cart is not None:
            line += f" | {cart[float(times[k])] * 100:24.1f} %"
        print(line)

        ax.plot(r, h_fv, "-", color=c, lw=1.6, alpha=0.7, label=f"FV t={times[k]:.2f}s")
        ax.plot(r, h_pinn, "--", color=c, lw=2, label=f"1D-radyal PINN t={times[k]:.2f}s")

    ax.axvline(R0, ls=":", color="gray", lw=1, label=f"R0={R0}m")
    ax.set(title="1D radyal PINN (--) vs radyal FV referansı (—)",
           xlabel="r (m)", ylabel="h (m)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "2d", f"radial_1d{args.tag}_vs_fv.png")
    fig.savefig(out, dpi=120)

    print("  " + "-" * (60 if cart is not None else 28))
    if cart is not None:
        print("  Beklenti: 1D-radyal PINN L2 ≤ 2D Kartezyen — simetri zorlandığı için")
        print("            (özellikle cephe + geç-zaman). Kelebek (asimetri) yok.")
    print(f"  ✓ Grafik: {out}")


if __name__ == "__main__":
    main()
