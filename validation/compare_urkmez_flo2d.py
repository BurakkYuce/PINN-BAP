"""Faz 2B doğrulama — Ürkmez 2D PINN vs FLO-2D maksimum derinlik rasterı.

`pinn_urkmez_2d.py` çıktısındaki maksimum derinlik haritasını (h_max), FLO-2D'nin
ürettiği maksimum derinlik rasterıyla HÜCRE-HÜCRE karşılaştırır ve hata haritası +
istatistik üretir. Referans benchmark: Haltaş, Tayfur & Elçi (2016) — FLO-2D için
bildirilen ~%14 derinlik hatası.

FLO-2D ÇIKTISI (kullanıcı üretir) — iki format desteklenir
----------------------------------------------------------
1) ESRI ASCII grid (.asc)  — FLO-2D'nin standart raster dışa aktarımı:
       ncols 200
       nrows 150
       xllcorner 0.0
       yllcorner 0.0
       cellsize 20.0
       NODATA_value -9999
       <nrows satır, her biri ncols değer; ÜST satır en büyük y>

2) CSV — üç kolon (başlık satırı opsiyonel):  x, y, max_depth

Dosya yolu: data/raw/urkmez_flo2d_maxdepth.asc  (veya .csv)
Ortam değişkeniyle de verilebilir:  FLO2D_PATH=/yol/dosya.asc

Çalıştırma:
    python validation/compare_urkmez_flo2d.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "2d")
DEFAULT_PATHS = [
    os.path.join(ROOT, "data", "raw", "urkmez_flo2d_maxdepth.asc"),
    os.path.join(ROOT, "data", "raw", "urkmez_flo2d_maxdepth.csv"),
]
BENCHMARK_PCT = 14.0   # Haltaş vd. 2016 — FLO-2D derinlik hatası


# ----------------------------------------------------------------------
# FLO-2D okuyucular
# ----------------------------------------------------------------------
def read_esri_ascii(path):
    """ESRI ASCII grid okur -> (x, y, Z) ; y ARTAN sırada, Z[j,i] = (y_j, x_i)."""
    header, values = {}, []
    keys = {"ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
            "yllcenter", "cellsize", "nodata_value"}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0].lower() in keys and len(parts) >= 2:
                header[parts[0].lower()] = float(parts[1])
            else:
                values.extend(float(v) for v in parts)

    for req in ("ncols", "nrows", "cellsize"):
        if req not in header:
            raise ValueError(f"ASCII grid başlığında '{req}' yok: {path}")
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    cs = header["cellsize"]
    nodata = header.get("nodata_value", -9999.0)

    if len(values) != ncols * nrows:
        raise ValueError(
            f"Veri sayısı başlıkla uyuşmuyor: {len(values)} != {ncols}×{nrows}")

    Z = np.array(values, dtype=float).reshape(nrows, ncols)
    Z[Z == nodata] = np.nan
    # ASCII grid ÜSTTEN alta yazılır -> y'yi artan yapmak için ters çevir
    Z = Z[::-1, :]

    if "xllcenter" in header:
        x0, y0 = header["xllcenter"], header["yllcenter"]
    else:
        x0 = header.get("xllcorner", 0.0) + cs / 2.0
        y0 = header.get("yllcorner", 0.0) + cs / 2.0
    x = x0 + cs * np.arange(ncols)
    y = y0 + cs * np.arange(nrows)
    return x, y, Z


def read_csv_points(path, target_x, target_y):
    """CSV (x,y,depth) okur ve hedef ızgaraya enterpole eder."""
    raw = np.genfromtxt(path, delimiter=",", names=None, dtype=float,
                        invalid_raise=False)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"CSV en az 3 kolon olmalı (x,y,depth): {path}")
    raw = raw[~np.isnan(raw).any(axis=1)]
    XX, YY = np.meshgrid(target_x, target_y)
    Z = griddata(raw[:, :2], raw[:, 2], (XX, YY), method="linear")
    return Z


def load_flo2d(path, target_x, target_y):
    """FLO-2D çıktısını PINN ızgarasına oturtulmuş dizi olarak döndürür."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_csv_points(path, target_x, target_y)

    x, y, Z = read_esri_ascii(path)
    interp = RegularGridInterpolator((y, x), Z, bounds_error=False, fill_value=np.nan)
    XX, YY = np.meshgrid(target_x, target_y)
    return interp(np.column_stack([YY.ravel(), XX.ravel()])).reshape(XX.shape)


def _resolve_path():
    env = os.environ.get("FLO2D_PATH")
    if env:
        return env if os.path.exists(env) else None
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    return None


def _missing_file_help():
    case = json.load(open(os.path.join(ROOT, "cases", "urkmez.json")))
    print("[compare_urkmez_flo2d] Faz 2B — FLO-2D karşılaştırması")
    print("=" * 70)
    print(f"  Vaka     : {case.get('name')}  ({case.get('dam_type_tr')})")
    print(f"  Referans : {case.get('reference')}")
    print("\n  ⚠ FLO-2D çıktısı bulunamadı. Aranan yollar:")
    for p in DEFAULT_PATHS:
        print(f"      {p}")
    print("    (veya FLO2D_PATH=/yol/dosya.asc ortam değişkeni)")
    print("\n  Beklenen format — ESRI ASCII grid (.asc):")
    print("      ncols / nrows / xllcorner / yllcorner / cellsize / NODATA_value")
    print("      ardından nrows satır × ncols değer (üst satır en büyük y)")
    print("  veya CSV: x, y, max_depth")
    print("\n  Dosya eklendiğinde bu betik hücre-hücre karşılaştırmayı ve hata")
    print("  haritasını otomatik üretir; kod TAMAMLANMIŞ durumdadır.")


# ----------------------------------------------------------------------
def main() -> None:
    pinn_npz = os.path.join(RESULTS, "urkmez_2d_prediction.npz")
    if not os.path.exists(pinn_npz):
        raise FileNotFoundError(
            f"{pinn_npz} yok.\n"
            "Önce: DDE_BACKEND=pytorch python src/pinn/pinn_urkmez_2d.py [--smoke]")

    P = np.load(pinn_npz)
    xs, ys, h_pinn = P["x"], P["y"], P["h_max"]

    flo_path = _resolve_path()
    if flo_path is None:
        _missing_file_help()
        return

    print("[compare_urkmez_flo2d] Faz 2B — Ürkmez: PINN vs FLO-2D")
    print("=" * 70)
    print(f"  FLO-2D dosyası : {flo_path}")
    h_flo = load_flo2d(flo_path, xs, ys)

    valid = np.isfinite(h_flo) & np.isfinite(h_pinn)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("Ortak geçerli hücre yok — FLO-2D alanı PINN alanıyla "
                         "örtüşmüyor olabilir (koordinat sistemi/extent kontrol et).")

    diff = h_pinn - h_flo
    d = diff[valid]
    ref = h_flo[valid]
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    bias = float(np.mean(d))
    denom = float(np.mean(np.abs(ref)))
    rel_pct = 100.0 * mae / denom if denom > 0 else float("nan")

    print(f"  Karşılaştırılan hücre : {n_valid} / {h_pinn.size} "
          f"({100.0 * n_valid / h_pinn.size:.0f}%)")
    print(f"  FLO-2D ort. derinlik  : {denom:.3f} m")
    print("-" * 70)
    print(f"  MAE   |h_PINN - h_FLO2D| : {mae:8.3f} m")
    print(f"  RMSE                     : {rmse:8.3f} m")
    print(f"  Bias  (PINN - FLO2D)     : {bias:+8.3f} m")
    print(f"  Bağıl hata (MAE/ort.)    : {rel_pct:8.1f} %   "
          f"(benchmark ≈ {BENCHMARK_PCT:.0f}%)")
    verdict = "✓ benchmark seviyesinde" if rel_pct <= BENCHMARK_PCT else "⚠ benchmark üstünde"
    print(f"  Değerlendirme            : {verdict}")

    # --- Haritalar ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    vmax = float(np.nanmax([np.nanmax(h_pinn), np.nanmax(h_flo)]))
    for ax, Z, title in ((axes[0], h_pinn, "PINN — maks derinlik"),
                         (axes[1], h_flo, "FLO-2D — maks derinlik")):
        im = ax.pcolormesh(xs, ys, Z, shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="h (m)")

    lim = float(np.nanpercentile(np.abs(diff[valid]), 98)) or 1.0
    im = axes[2].pcolormesh(xs, ys, np.where(valid, diff, np.nan), shading="auto",
                            cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set(title=f"Fark (PINN − FLO-2D)\nMAE={mae:.2f} m, bağıl={rel_pct:.1f}%",
                xlabel="x (m)", ylabel="y (m)")
    axes[2].set_aspect("equal")
    fig.colorbar(im, ax=axes[2], label="Δh (m)")

    fig.suptitle("Faz 2B — Ürkmez Barajı: 2D PINN vs FLO-2D maksimum derinlik")
    fig.tight_layout()
    out = os.path.join(RESULTS, "urkmez_pinn_vs_flo2d.png")
    fig.savefig(out, dpi=120)
    print(f"\n  ✓ Hata haritası: {out}")
    print("\n  ⚠ Hatırlatma: taban DÜZ+sabit eğim kabul edildi (gerçek DEM yok).")
    print("    Gerçek topografya eklenene kadar bu karşılaştırma NİTELİKSELDİR.")


if __name__ == "__main__":
    main()
