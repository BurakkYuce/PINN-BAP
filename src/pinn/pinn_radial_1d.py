"""Faz 2A (radyal) — 1D radyal (r,t) dam-break PINN (DeepXDE, PyTorch backend).

2D Kartezyen baseline'a (pinn_baseline_2d.py) PARALEL, ona DOKUNMAYAN modül.
Eksenel simetriyi denklem düzeyinde zorlar (silindirik SWE, 1/r kaynak):
  * kelebek / açısal asimetri TANIM GEREĞİ imkânsız,
  * tüm collocation bütçesi (r,t)'ye gider → cephe + merkez daha keskin,
  * 1D+zaman olduğundan CPU'da dakikalar (GPU gerekmez).

Veri YOK: ağ yalnızca radyal Saint-Venant + radyal IC + far-field BC'den öğrenir.

Simetri (r=0) çıktı dönüşümüyle KESİN sağlanır:  u = tanh(r/L) * u_raw  →  u(0,t)=0
ve 1/r kaynağındaki u/r sınırlı kalır. h pozitifliği softplus ile.

Çalıştırma:
    DDE_BACKEND=pytorch python src/pinn/pinn_radial_1d.py            # tam eğitim
    DDE_BACKEND=pytorch python src/pinn/pinn_radial_1d.py --smoke    # hızlı test

Çıktılar (results/2d/):
    radial_1d_model-*.pt
    radial_1d_prediction.npz      (compare_radial_1d.py okur)
    radial_1d_loss.png
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import deepxde as dde  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from physics.saint_venant_radial import make_pde_radial  # noqa: E402

# Radyal koşu için eğitim ayarları (1D ucuz; config'teki 2D sayıları kullanmıyoruz).
TRAIN = {
    "full":  dict(adam_iters=20000, lbfgs=True,  num_domain=6000,
                  num_boundary=400, num_initial=1200),
    "smoke": dict(adam_iters=2500,  lbfgs=False, num_domain=2000,
                  num_boundary=200, num_initial=600),
}
L_SYM = 0.5   # simetri uzunluk ölçeği: u = tanh(r/L_SYM)*u_raw → u(0,t)=0


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges_2d.yaml"), "r") as f:
        return yaml.safe_load(f)


def _geometry(cfg):
    """Radyal alan: r ∈ [0, R_max], t ∈ [0, T]. R_max = merkeze en yakın kenar."""
    d = cfg["baseline_2d"]["domain"]
    ric = cfg["baseline_2d"]["radial_ic"]
    R_max = min(ric["x_c"] - d["x_min"], d["x_max"] - ric["x_c"],
                ric["y_c"] - d["y_min"], d["y_max"] - ric["y_c"])
    T = d["T"]
    return R_max, T


def build_model(cfg: dict, train_cfg: dict,
                nu: float | None = None, eps: float | None = None) -> dde.Model:
    g = cfg["physics"]["g"]
    n = cfg["physics"]["n"]
    nu = cfg["physics"]["nu"] if nu is None else nu
    ric = cfg["baseline_2d"]["radial_ic"]
    R0, hl, hr = ric["R0"], ric["hl"], ric["hr"]
    eps = ric["eps"] if eps is None else eps
    R_max, T = _geometry(cfg)

    geom = dde.geometry.Interval(0.0, R_max)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde = make_pde_radial(g=g, n=n, S0=0.0, nu=nu)

    # --- Radyal yumuşatılmış başlangıç koşulu (2D ile aynı profil) ---
    def ic_h_func(X):
        r = X[:, 0:1]
        return hr + 0.5 * (hl - hr) * (1.0 - np.tanh((r - R0) / eps))

    def zero_func(X):
        return np.zeros((X.shape[0], 1))

    ic_h = dde.icbc.IC(geomtime, ic_h_func, lambda _, on_i: on_i, component=0)
    ic_u = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=1)

    # --- Far-field (r = R_max) Dirichlet: dalga T süresince ulaşmaz → h=hr, u=0 ---
    def on_far(X, on_boundary):
        return on_boundary and np.isclose(X[0], R_max)

    bc_h = dde.icbc.DirichletBC(geomtime, lambda X: np.full((X.shape[0], 1), hr),
                                on_far, component=0)
    bc_u = dde.icbc.DirichletBC(geomtime, zero_func, on_far, component=1)
    # r=0 simetrisi (u=0) çıktı dönüşümünde KESİN sağlandığı için ayrı BC gerekmez.

    data = dde.data.TimePDE(
        geomtime, pde,
        [ic_h, ic_u, bc_h, bc_u],
        num_domain=train_cfg["num_domain"],
        num_boundary=train_cfg["num_boundary"],
        num_initial=train_cfg["num_initial"],
    )

    layers = cfg["training"]["layers"]
    activation = cfg["training"]["activation"]
    net = dde.nn.FNN([2] + layers + [2], activation, "Glorot normal")

    # Girdi normalizasyonu ([-1,1]).
    def feature_transform(X):
        rs = 2.0 * X[:, 0:1] / R_max - 1.0
        ts = 2.0 * X[:, 1:2] / T - 1.0
        return torch.cat([rs, ts], dim=1)

    net.apply_feature_transform(feature_transform)

    # Çıktı dönüşümü: h>0 (softplus), u(0,t)=0 (tanh(r/L) çarpanı → 1/r sınırlı).
    def output_transform(X, Y):
        r = X[:, 0:1]
        h = torch.nn.functional.softplus(Y[:, 0:1])
        u = torch.tanh(r / L_SYM) * Y[:, 1:2]
        return torch.cat([h, u], dim=1)

    net.apply_output_transform(output_transform)

    return dde.Model(data, net)


def _save_prediction(model, cfg, results_dir, tag: str = ""):
    ric = cfg["baseline_2d"]["radial_ic"]
    R_max, T = _geometry(cfg)
    times = np.array(cfg["training"].get("predict_times", [T]), dtype=float)

    r = np.linspace(0.0, R_max, 400)
    H = np.zeros((len(times), r.size))
    U = np.zeros((len(times), r.size))
    flat = np.column_stack([r, np.zeros_like(r)])
    for k, t in enumerate(times):
        flat[:, 1] = t
        pred = model.predict(flat)
        H[k] = pred[:, 0]
        U[k] = pred[:, 1]

    out = os.path.join(results_dir, f"radial_1d{tag}_prediction.npz")
    np.savez(out, r=r, t=times, h=H, u=U,
             R0=ric["R0"], hl=ric["hl"], hr=ric["hr"], g=cfg["physics"]["g"],
             R_max=R_max)
    return out


def train(smoke: bool = False, nu: float | None = None,
          eps: float | None = None, tag: str = "") -> None:
    cfg = load_config()
    train_cfg = TRAIN["smoke" if smoke else "full"]
    results_dir = os.path.join(ROOT, "results", "2d")
    os.makedirs(results_dir, exist_ok=True)

    if nu is not None or eps is not None:
        print(f"[pinn_radial_1d] override: nu={nu}, eps={eps}, tag='{tag}'")
    model = build_model(cfg, train_cfg, nu=nu, eps=eps)

    # Loss sırası: pde(2) + ic_h + ic_u + bc_h + bc_u = 6
    loss_weights = [1.0, 1.0, 10.0, 10.0, 10.0, 10.0]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    print(f"[pinn_radial_1d] Adam ({train_cfg['adam_iters']} iter)...")
    losshistory, _ = model.train(iterations=train_cfg["adam_iters"], display_every=500)

    if train_cfg["lbfgs"]:
        print("[pinn_radial_1d] L-BFGS ince ayar...")
        model.compile("L-BFGS")
        losshistory, _ = model.train()

    model.save(os.path.join(results_dir, f"radial_1d{tag}_model"))
    out = _save_prediction(model, cfg, results_dir, tag=tag)

    try:
        dde.utils.plot_loss_history(losshistory)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(results_dir, f"radial_1d{tag}_loss.png"), dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"[pinn_radial_1d] loss grafiği atlandı: {exc}")

    print(f"[pinn_radial_1d] ✓ Tamamlandı. Tahmin: {out}")
    if smoke:
        print("[pinn_radial_1d] (smoke modu — convergence beklenmez; tam eğitim "
              "CPU'da birkaç dakika.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz 2A 1D radyal dam-break PINN")
    parser.add_argument("--smoke", action="store_true",
                        help="Hızlı uçtan uca test (az iterasyon, L-BFGS yok)")
    parser.add_argument("--nu", type=float, default=None,
                        help="Yapay viskoziteyi config yerine bu değere ayarla (şok keskinleştirme)")
    parser.add_argument("--eps", type=float, default=None,
                        help="IC yumuşatma genişliğini config yerine bu değere ayarla")
    parser.add_argument("--tag", type=str, default="",
                        help="Çıktı dosya eki (ör. '_lownu') — baseline'ı ezmemek için")
    args = parser.parse_args()
    train(smoke=args.smoke, nu=args.nu, eps=args.eps, tag=args.tag)
