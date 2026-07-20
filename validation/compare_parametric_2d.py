"""Faz 2C doğrulama — parametrik 2D PINN fizik testleri.

Parametrik modelin doğrulanması, tek-senaryo modellerinden FARKLIDIR: karşılaştırılacak
tek bir analitik çözüm yoktur (parametre uzayının her noktası ayrı bir problem).
Bu yüzden çözümün SAĞLAMASI GEREKEN fiziksel değişmezler test edilir:

  TEST 1 — Kütle korunumu.
      Alan T süresince kapalıdır (dalga uzak sınıra ulaşmaz). Toplam su hacmi
      ∫h dA zamanla değişmemeli. Parametreden bağımsız, güçlü bir testtir.

  TEST 2 — Parametre duyarlılığı (fiziksel tekdüzelik).
      Daha GENİŞ breach (B↑) veya daha HIZLI kırılma (t_f↓) => T anına kadar
      mansaba daha çok su geçmeli. Model bu yönelimi öğrenmemişse parametre
      bağımlılığı sahtedir.

  TEST 3 — Senaryo haritaları.
      Temsili (B,t_f,h0,n) noktalarında h(x,y) görselleştirmesi.

NOT: "tam genişlik + ani açılma -> 1D Stoker" limiti bilerek KULLANILMADI: breach
kademeli açıldığı için açıklık kenarlarının etkisi merkez hattına ~B/(2·sqrt(g·h0))
sürede ulaşır (tipik olarak ~1 s) ve temiz bir 1D pencere kalmaz.

Çalıştırma:
    DDE_BACKEND=pytorch python validation/compare_parametric_2d.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pinn.pinn_parametric_2d import (  # noqa: E402
    COL, INPUT_DIM, PARAM_KEYS, build_model, load_config, _bounds)

RESULTS = os.path.join(ROOT, "results", "2d")


def _latest_checkpoint():
    cands = glob.glob(os.path.join(RESULTS, "parametric_2d_model-*.pt"))
    if not cands:
        return None
    def step(p):
        m = re.search(r"-(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(cands, key=step)


def _grid(cfg, nx=140, ny=70):
    d = cfg["parametric_2d"]["domain"]
    xs = np.linspace(d["x_min"], d["x_max"], nx)
    ys = np.linspace(d["y_min"], d["y_max"], ny)
    return xs, ys, np.meshgrid(xs, ys)


def _predict(model, XX, YY, t, params):
    """params: dict(B,t_f,h0,n) -> h,u,v ızgarada."""
    flat = np.zeros((XX.size, INPUT_DIM))
    flat[:, COL["x"]] = XX.ravel()
    flat[:, COL["y"]] = YY.ravel()
    for k in PARAM_KEYS:
        flat[:, COL[k]] = params[k]
    flat[:, COL["t"]] = t
    p = model.predict(flat)
    shape = XX.shape
    return (p[:, 0].reshape(shape), p[:, 1].reshape(shape), p[:, 2].reshape(shape))


def _volume(h, xs, ys, x_from=None):
    """∫h dA (trapez). x_from verilirse yalnız x>x_from bölgesi."""
    if x_from is not None:
        mask = xs >= x_from
        return np.trapz(np.trapz(h[:, mask], xs[mask], axis=1), ys)
    return np.trapz(np.trapz(h, xs, axis=1), ys)


def main() -> None:
    cfg = load_config()
    pc = cfg["parametric_2d"]
    pr = pc["param_ranges"]
    lower, upper, T = _bounds(cfg)
    x_dam = pc["dam"]["x_dam"]

    ckpt = _latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(
            f"{RESULTS}/parametric_2d_model-*.pt yok.\n"
            "Önce: DDE_BACKEND=pytorch python src/pinn/pinn_parametric_2d.py --smoke")

    # Ağ mimarisi train_cfg'den bağımsız; tahmin için küçük veri kurulumu yeterli.
    model = build_model(cfg, pc["training"]["smoke"])
    model.compile("adam", lr=1e-3)
    model.restore(ckpt, verbose=0)
    print(f"Faz 2C doğrulama — model: {os.path.basename(ckpt)}")
    print("=" * 68)

    xs, ys, (XX, YY) = _grid(cfg)
    mid = {k: 0.5 * (pr[k]["min"] + pr[k]["max"]) for k in PARAM_KEYS}

    # ---------------- TEST 1: kütle korunumu ----------------
    print("\nTEST 1 — Kütle korunumu (∫h dA, t=0 vs t=T)")
    print("-" * 68)
    print(f"  {'senaryo':>22} | {'V(0) [m³]':>13} | {'V(T) [m³]':>13} | {'sapma':>8}")
    rng = np.random.default_rng(0)
    test_pts = [
        ("orta", mid),
        ("dar-yavaş", dict(B=pr["B"]["min"], t_f=pr["t_f"]["max"],
                           h0=pr["h0"]["min"], n=0.04)),
        ("geniş-hızlı", dict(B=pr["B"]["max"], t_f=pr["t_f"]["min"],
                             h0=pr["h0"]["max"], n=0.04)),
    ]
    for _ in range(2):  # rastgele iki nokta daha
        p = {k: float(rng.uniform(pr[k]["min"], pr[k]["max"])) for k in PARAM_KEYS}
        test_pts.append(("rastgele", p))

    drifts = []
    for name, p in test_pts:
        h0g, _, _ = _predict(model, XX, YY, 0.0, p)
        hTg, _, _ = _predict(model, XX, YY, T, p)
        V0, VT = _volume(h0g, xs, ys), _volume(hTg, xs, ys)
        drift = 100.0 * (VT - V0) / V0
        drifts.append(abs(drift))
        print(f"  {name:>22} | {V0:13.4g} | {VT:13.4g} | {drift:+7.1f}%")
    print(f"  {'→ ortalama |sapma|':>22} : {np.mean(drifts):.1f}%   "
          f"(iyi < %5, kabul edilebilir < %15)")

    # ---------------- TEST 2: parametre duyarlılığı ----------------
    print("\nTEST 2 — Parametre duyarlılığı (mansaba geçen hacim, t=T)")
    print("-" * 68)

    def downstream_volume(p):
        h, _, _ = _predict(model, XX, YY, T, p)
        return _volume(h, xs, ys, x_from=x_dam)

    sweeps = {}
    Bs = np.linspace(pr["B"]["min"], pr["B"]["max"], 5)
    volB = [downstream_volume({**mid, "B": float(b)}) for b in Bs]
    sweeps["B"] = (Bs, volB, "artmalı")
    tfs = np.linspace(pr["t_f"]["min"], pr["t_f"]["max"], 5)
    volT = [downstream_volume({**mid, "t_f": float(tf)}) for tf in tfs]
    sweeps["t_f"] = (tfs, volT, "azalmalı")

    def monotonic(v, increasing):
        d = np.diff(v)
        frac = np.mean(d > 0) if increasing else np.mean(d < 0)
        return frac

    for key, (vals, vols, beklenti) in sweeps.items():
        inc = (beklenti == "artmalı")
        frac = monotonic(np.array(vols), inc)
        ok = "✓" if frac >= 0.75 else "✗"
        print(f"  {key:>5} sweep ({beklenti:>8}): " +
              "  ".join(f"{v:.3g}" for v in vols))
        print(f"        tekdüzelik: {frac * 100:.0f}% adım doğru yönde  {ok}")

    # ---------------- TEST 3: senaryo haritaları ----------------
    print("\nTEST 3 — Senaryo haritaları")
    print("-" * 68)
    scen = [("dar-yavaş", dict(B=pr["B"]["min"], t_f=pr["t_f"]["max"],
                               h0=pr["h0"]["min"], n=0.04)),
            ("orta", mid),
            ("geniş-hızlı", dict(B=pr["B"]["max"], t_f=pr["t_f"]["min"],
                                 h0=pr["h0"]["max"], n=0.04))]
    times = [0.25 * T, 0.5 * T, T]
    fig, axes = plt.subplots(len(scen), len(times),
                             figsize=(4.6 * len(times), 2.9 * len(scen)), squeeze=False)
    for i, (name, p) in enumerate(scen):
        for j, t in enumerate(times):
            h, _, _ = _predict(model, XX, YY, t, p)
            ax = axes[i][j]
            im = ax.pcolormesh(xs, ys, h, shading="auto", cmap="viridis",
                               vmin=0.0, vmax=max(p["h0"] * 0.6, 1e-3))
            ax.axvline(x_dam, color="w", ls="--", lw=.8)
            ax.set_title(f"{name} · t={t:.0f}s", fontsize=9)
            ax.set_aspect("equal")
            if j == 0:
                ax.set_ylabel(f"B={p['B']:.0f}m\nt_f={p['t_f']:.0f}s\nh0={p['h0']:.0f}m",
                              fontsize=7)
            fig.colorbar(im, ax=ax, label="h (m)")
    fig.suptitle("Faz 2C — parametrik 2D PINN: farklı breach senaryoları", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS, "parametric_2d_scenarios.png")
    fig.savefig(out, dpi=115)
    print(f"  ✓ {out}")

    # duyarlılık grafiği
    fig2, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (key, (vals, vols, beklenti)) in zip(axs, sweeps.items()):
        ax.plot(vals, vols, "o-", color="#12657A", lw=1.8)
        ax.set(xlabel=f"{key}", ylabel="mansap hacmi (m³)",
               title=f"{key} artarken hacim {beklenti}")
        ax.grid(alpha=.3)
    fig2.tight_layout()
    out2 = os.path.join(RESULTS, "parametric_2d_sensitivity.png")
    fig2.savefig(out2, dpi=115)
    print(f"  ✓ {out2}")

    print("\n" + "=" * 68)
    print("Yorum: TEST 1 çözümün fiziksel tutarlılığını, TEST 2 parametre")
    print("bağımlılığının GERÇEK olduğunu sınar. Smoke eğitiminde her ikisinin de")
    print("başarısız olması normaldir — tam eğitim (GPU) sonrası tekrar koşulmalı.")


if __name__ == "__main__":
    main()
