"""Faz 1 doğrulama — PINN tahmini vs Stoker analitik çözümü.

`pinn_baseline.py` çalıştırıldıktan sonra üretilen results/baseline_prediction.npz
dosyasını okur, aynı t anında Stoker referans çözümünü hesaplar, h(x) ve u(x)
profillerini üst üste çizer ve bağıl L2 hatasını raporlar.

Çalıştırma:
    python validation/compare_baseline.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stoker_analytic import stoker  # noqa: E402  (aynı klasör, sys.path[0])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def main() -> None:
    npz_path = os.path.join(ROOT, "results", "baseline_prediction.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} bulunamadı. Önce: python src/pinn/pinn_baseline.py [--smoke]")

    data = np.load(npz_path)
    x = data["x"]
    t = float(data["t"])
    h_pred, u_pred = data["h"], data["u"]
    hl, hr, x0, g = float(data["hl"]), float(data["hr"]), float(data["x0"]), float(data["g"])

    h_ref, u_ref = stoker(x, t, hl=hl, hr=hr, x0=x0, g=g)

    err_h = rel_l2(h_pred, h_ref)
    err_u = rel_l2(u_pred, u_ref)

    print("Faz 1 — PINN vs Stoker doğrulaması")
    print("-" * 40)
    print(f"  t = {t} s,  hl={hl}, hr={hr}")
    print(f"  bağıl L2 hata  h: {err_h * 100:6.2f} %")
    print(f"  bağıl L2 hata  u: {err_u * 100:6.2f} %")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, h_ref, "k-", lw=2, label="Stoker (analitik)")
    ax[0].plot(x, h_pred, "r--", lw=2, label="PINN")
    ax[0].set(title=f"Su derinliği h (t={t}s) — hata {err_h*100:.1f}%",
              xlabel="x (m)", ylabel="h (m)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(x, u_ref, "k-", lw=2, label="Stoker (analitik)")
    ax[1].plot(x, u_pred, "r--", lw=2, label="PINN")
    ax[1].set(title=f"Hız u (t={t}s) — hata {err_u*100:.1f}%",
              xlabel="x (m)", ylabel="u (m/s)")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle("Faz 1: Parametrik-olmayan dam-break PINN doğrulaması")
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "baseline_vs_stoker.png")
    fig.savefig(out, dpi=120)
    print(f"  ✓ Grafik kaydedildi: {out}")


if __name__ == "__main__":
    main()
