"""Faz 2A — 2D radyal dam-break PINN (DeepXDE, PyTorch backend).

Veri YOK: ağ yalnızca 2D Saint-Venant denklemi + radyal IC + far-field BC'den öğrenir.
Merkezde R0 yarıçaplı, hl derinliğinde su sütunu dışa doğru yayılır.

Çalıştırma:
    DDE_BACKEND=pytorch python src/pinn/pinn_baseline_2d.py            # tam eğitim
    DDE_BACKEND=pytorch python src/pinn/pinn_baseline_2d.py --smoke    # hızlı test

Çıktılar (results/2d/):
    baseline_2d_model-*.pt          model
    baseline_2d_prediction.npz      grid tahmini (heatmap + radyal kesit için)
    baseline_2d_loss.png            loss geçmişi
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
from physics.saint_venant_2d import make_pde_2d  # noqa: E402


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges_2d.yaml"), "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, train_cfg: dict) -> dde.Model:
    g = cfg["physics"]["g"]
    n = cfg["physics"]["n"]
    S0x, S0y = cfg["physics"]["S0x"], cfg["physics"]["S0y"]
    nu = cfg["physics"]["nu"]

    d = cfg["baseline_2d"]["domain"]
    x_min, x_max = d["x_min"], d["x_max"]
    y_min, y_max = d["y_min"], d["y_max"]
    T = d["T"]
    ric = cfg["baseline_2d"]["radial_ic"]
    x_c, y_c, R0, hl, hr, eps = (ric["x_c"], ric["y_c"], ric["R0"],
                                 ric["hl"], ric["hr"], ric["eps"])

    geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde = make_pde_2d(g=g, n=n, S0x=S0x, S0y=S0y, nu=nu)

    # --- Radyal yumuşatılmış başlangıç koşulu ---
    def ic_h_func(X):
        x, y = X[:, 0:1], X[:, 1:2]
        r = np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2)
        return hr + 0.5 * (hl - hr) * (1.0 - np.tanh((r - R0) / eps))

    def zero_func(X):
        return np.zeros((X.shape[0], 1))

    ic_h = dde.icbc.IC(geomtime, ic_h_func, lambda _, on_i: on_i, component=0)
    ic_u = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=1)
    ic_v = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=2)

    # --- Far-field sınır koşulu (Dirichlet) ---
    # Alan, dalga T süresince kenarlara ulaşmayacak şekilde seçildi; kenarda
    # dış değerler sabit (h=hr, u=v=0). Bu, dalga ulaşmadığı sürece yansımasızdır.
    def on_bnd(X, on_boundary):
        return on_boundary

    bc_h = dde.icbc.DirichletBC(geomtime, lambda X: np.full((X.shape[0], 1), hr),
                                on_bnd, component=0)
    bc_u = dde.icbc.DirichletBC(geomtime, zero_func, on_bnd, component=1)
    bc_v = dde.icbc.DirichletBC(geomtime, zero_func, on_bnd, component=2)

    data = dde.data.TimePDE(
        geomtime, pde,
        [ic_h, ic_u, ic_v, bc_h, bc_u, bc_v],
        num_domain=train_cfg["num_domain"],
        num_boundary=train_cfg["num_boundary"],
        num_initial=train_cfg["num_initial"],
    )

    layers = cfg["training"]["layers"]
    activation = cfg["training"]["activation"]
    net = dde.nn.FNN([3] + layers + [3], activation, "Glorot normal")

    # Girdi normalizasyonu ([-1,1]) — koşullanmayı iyileştirir.
    def feature_transform(X):
        xs = 2.0 * (X[:, 0:1] - x_min) / (x_max - x_min) - 1.0
        ys = 2.0 * (X[:, 1:2] - y_min) / (y_max - y_min) - 1.0
        tsn = 2.0 * X[:, 2:3] / T - 1.0
        return torch.cat([xs, ys, tsn], dim=1)

    net.apply_feature_transform(feature_transform)

    # h > 0 pozitifliği (2D'de şart: negatif derinlik h^(4/3)'ü patlatır).
    def output_transform(X, Y):
        h = torch.nn.functional.softplus(Y[:, 0:1])
        return torch.cat([h, Y[:, 1:2], Y[:, 2:3]], dim=1)

    net.apply_output_transform(output_transform)

    return dde.Model(data, net)


def _save_prediction(model, cfg, results_dir):
    d = cfg["baseline_2d"]["domain"]
    ric = cfg["baseline_2d"]["radial_ic"]
    times = np.array(cfg["training"].get("predict_times", [d["T"]]), dtype=float)

    nx = ny = 120
    xs = np.linspace(d["x_min"], d["x_max"], nx)
    ys = np.linspace(d["y_min"], d["y_max"], ny)
    XX, YY = np.meshgrid(xs, ys)
    H = np.zeros((len(times), ny, nx))
    U = np.zeros((len(times), ny, nx))
    V = np.zeros((len(times), ny, nx))
    flat = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    for k, t in enumerate(times):
        flat[:, 2] = t
        pred = model.predict(flat)
        H[k] = pred[:, 0].reshape(ny, nx)
        U[k] = pred[:, 1].reshape(ny, nx)
        V[k] = pred[:, 2].reshape(ny, nx)

    out = os.path.join(results_dir, "baseline_2d_prediction.npz")
    np.savez(out, x=xs, y=ys, t=times, h=H, u=U, v=V,
             x_c=ric["x_c"], y_c=ric["y_c"], R0=ric["R0"],
             hl=ric["hl"], hr=ric["hr"], g=cfg["physics"]["g"])
    return out


def train(smoke: bool = False) -> None:
    cfg = load_config()
    train_cfg = cfg["training"]["smoke" if smoke else "full"]
    results_dir = os.path.join(ROOT, "results", "2d")

    model = build_model(cfg, train_cfg)

    # Loss sırası: pde(3) + ic_h + ic_u + ic_v + bc_h + bc_u + bc_v = 9
    loss_weights = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    print(f"[pinn_baseline_2d] Adam ({train_cfg['adam_iters']} iter)...")
    losshistory, _ = model.train(iterations=train_cfg["adam_iters"], display_every=500)

    if train_cfg["lbfgs"]:
        print("[pinn_baseline_2d] L-BFGS ince ayar...")
        model.compile("L-BFGS")
        losshistory, _ = model.train()

    model.save(os.path.join(results_dir, "baseline_2d_model"))
    out = _save_prediction(model, cfg, results_dir)

    try:
        dde.utils.plot_loss_history(losshistory)
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(results_dir, "baseline_2d_loss.png"), dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"[pinn_baseline_2d] loss grafiği atlandı: {exc}")

    print(f"[pinn_baseline_2d] ✓ Tamamlandı. Tahmin: {out}")
    if smoke:
        print("[pinn_baseline_2d] (smoke modu — convergence beklenmez; tam eğitim "
              "için GPU/Colab önerilir, bkz. README.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz 2A 2D radyal dam-break PINN")
    parser.add_argument("--smoke", action="store_true",
                        help="Hızlı uçtan uca test (az iterasyon, L-BFGS yok)")
    args = parser.parse_args()
    train(smoke=args.smoke)
