"""Faz 1 — Tek senaryo 1D dam-break PINN (DeepXDE, PyTorch backend).

Veri YOK: ağ yalnızca Saint-Venant denklemi + başlangıç/sınır koşullarından öğrenir.
İki senaryo desteklenir (config'ten okunur):

  * baseline     : idealize, sürtünmesiz -> Stoker analitik ile karşılaştırma (Faz 1)
  * experimental : Vosoughi vd. (2021) flume (H=0.30 m, ıslak mansap, sürtünmeli)
                   -> gerçek laboratuvar ölçümleriyle karşılaştırma (Katman 2)

Çalıştırma:
    DDE_BACKEND=pytorch python src/pinn/pinn_baseline.py                      # baseline tam
    DDE_BACKEND=pytorch python src/pinn/pinn_baseline.py --smoke              # baseline hızlı
    DDE_BACKEND=pytorch python src/pinn/pinn_baseline.py --scenario experimental

Çıktılar (results/):
    {scenario}_model-*.pt        eğitilmiş model
    {scenario}_prediction.npz    tahmin (baseline: tek t; experimental: çok t)
    {scenario}_loss.png          loss geçmişi
"""
from __future__ import annotations

import argparse
import os
import sys

# Backend, deepxde import edilmeden ÖNCE sabitlenir.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import deepxde as dde  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from physics.saint_venant import make_pde  # noqa: E402


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges.yaml"), "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, train_cfg: dict, scenario: str) -> dde.Model:
    """Seçili senaryo için geometri + IC/BC + ağ + model kurulumu."""
    g = cfg["physics"]["g"]
    S0 = cfg["physics"]["S0"]
    scen = cfg[scenario]
    n = scen["manning_n"]
    nu = scen.get("nu", cfg["physics"]["nu"])

    d = scen["domain"]
    x_min, x_max, x0, T = d["x_min"], d["x_max"], d["x0"], d["T"]
    ic = scen["initial"]
    hl, hr, eps = ic["hl"], ic["hr"], ic["eps"]

    geom = dde.geometry.Interval(x_min, x_max)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde = make_pde(g=g, n=n, S0=S0, nu=nu)

    # --- Başlangıç koşulu (t=0): tanh ile yumuşatılmış basamak ---
    def ic_h_func(X):
        x = X[:, 0:1]
        return hr + 0.5 * (hl - hr) * (1.0 - np.tanh((x - x0) / eps))

    def ic_u_func(X):
        return np.zeros((X.shape[0], 1))

    ic_h = dde.icbc.IC(geomtime, ic_h_func, lambda _, on_initial: on_initial, component=0)
    ic_u = dde.icbc.IC(geomtime, ic_u_func, lambda _, on_initial: on_initial, component=1)

    # --- Sınır koşulları (Dirichlet, far-field) ---
    # Alan/zaman penceresi dalga sınıra ulaşmayacak şekilde seçildi; uzak-alan
    # değerleri sabit (sol: hl, sağ: hr, her ikisinde u=0).
    def on_left(X, on_boundary):
        return on_boundary and np.isclose(X[0], x_min)

    def on_right(X, on_boundary):
        return on_boundary and np.isclose(X[0], x_max)

    bc_h_left = dde.icbc.DirichletBC(
        geomtime, lambda X: np.full((X.shape[0], 1), hl), on_left, component=0)
    bc_h_right = dde.icbc.DirichletBC(
        geomtime, lambda X: np.full((X.shape[0], 1), hr), on_right, component=0)
    bc_u = dde.icbc.DirichletBC(
        geomtime, lambda X: np.zeros((X.shape[0], 1)),
        lambda X, on_boundary: on_boundary, component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_h, ic_u, bc_h_left, bc_h_right, bc_u],
        num_domain=train_cfg["num_domain"],
        num_boundary=train_cfg["num_boundary"],
        num_initial=train_cfg["num_initial"],
    )

    layers = cfg["training"]["layers"]
    activation = cfg["training"]["activation"]
    net = dde.nn.FNN([2] + layers + [2], activation, "Glorot normal")

    # Girdi normalizasyonu ([-1,1]) — özellikle fiziksel-birimli (experimental)
    # alanlarda ağ koşullanmasını iyileştirir; PDE türevleri fiziksel x'e göredir.
    def feature_transform(X):
        x_s = 2.0 * (X[:, 0:1] - x_min) / (x_max - x_min) - 1.0
        t_s = 2.0 * X[:, 1:2] / T - 1.0
        return torch.cat([x_s, t_s], dim=1)

    net.apply_feature_transform(feature_transform)

    # h > 0 pozitifliğini garanti et (softplus); u serbest.
    def output_transform(X, y):
        h = torch.nn.functional.softplus(y[:, 0:1])
        u = y[:, 1:2]
        return torch.cat([h, u], dim=1)

    net.apply_output_transform(output_transform)

    return dde.Model(data, net)


def _save_prediction(model, cfg, scenario, results_dir):
    """Senaryoya göre tahmini npz'e kaydeder."""
    scen = cfg[scenario]
    d = scen["domain"]
    ic = scen["initial"]
    xs = np.linspace(d["x_min"], d["x_max"], 400)
    g = cfg["physics"]["g"]
    out = os.path.join(results_dir, f"{scenario}_prediction.npz")

    if scenario == "experimental":
        times = np.array(scen.get("predict_times", [d["T"]]), dtype=float)
        H = np.zeros((len(times), len(xs)))
        U = np.zeros((len(times), len(xs)))
        for k, t in enumerate(times):
            pred = model.predict(np.column_stack([xs, np.full_like(xs, t)]))
            H[k], U[k] = pred[:, 0], pred[:, 1]
        np.savez(out, x=xs, t=times, h=H, u=U,
                 hl=ic["hl"], hr=ic["hr"], x0=d["x0"], g=g)
    else:
        T = d["T"]
        pred = model.predict(np.column_stack([xs, np.full_like(xs, T)]))
        np.savez(out, x=xs, t=T, h=pred[:, 0], u=pred[:, 1],
                 hl=ic["hl"], hr=ic["hr"], x0=d["x0"], g=g)
    return out


def train(scenario: str = "baseline", smoke: bool = False) -> None:
    cfg = load_config()
    train_cfg = cfg["training"][scenario]["smoke" if smoke else "full"]
    results_dir = os.path.join(ROOT, "results")

    model = build_model(cfg, train_cfg, scenario)

    # Loss terim sırası: pde(2) + ic_h + ic_u + bc_h_left + bc_h_right + bc_u = 7
    loss_weights = [1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    print(f"[pinn_baseline:{scenario}] Adam ({train_cfg['adam_iters']} iter)...")
    losshistory, _ = model.train(iterations=train_cfg["adam_iters"], display_every=500)

    if train_cfg["lbfgs"]:
        print(f"[pinn_baseline:{scenario}] L-BFGS ince ayar...")
        model.compile("L-BFGS")
        losshistory, _ = model.train()

    model.save(os.path.join(results_dir, f"{scenario}_model"))
    out = _save_prediction(model, cfg, scenario, results_dir)

    try:
        dde.utils.plot_loss_history(losshistory)
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(results_dir, f"{scenario}_loss.png"), dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001 — grafik opsiyonel
        print(f"[pinn_baseline:{scenario}] loss grafiği atlandı: {exc}")

    print(f"[pinn_baseline:{scenario}] ✓ Tamamlandı. Tahmin: {out}")
    if smoke:
        print("[pinn_baseline] (smoke modu — convergence beklenmez; tam eğitim önerilir.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz 1 dam-break PINN (DeepXDE)")
    parser.add_argument("--scenario", default="baseline",
                        choices=["baseline", "experimental"],
                        help="Senaryo (config'teki blok)")
    parser.add_argument("--smoke", action="store_true",
                        help="Hızlı uçtan uca test (az iterasyon, L-BFGS yok)")
    args = parser.parse_args()
    train(scenario=args.scenario, smoke=args.smoke)
