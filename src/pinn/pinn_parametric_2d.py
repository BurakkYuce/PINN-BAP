"""Faz 2C — Parametrik 2D dam-break PINN (h, u, v = f(x, y, t, B, t_f, h0, n)).

Projenin asıl frontier'ı: ağ tek bir senaryoyu değil bir senaryo AİLESİNİ öğrenir.
Bir kez eğitilir; sonra her (B, t_f, h0, n) dörtlüsü için çözüm anında okunur.

KURULUM
-------
Rezervuar ALAN İÇİNDEDİR: x < x_dam bölgesi h0 derinliğinde su tutar, mansap ise
ince bir ıslak tabaka (h_min). Baraj gövdesi x = x_dam çizgisindedir ve ortasında
ZAMANLA BÜYÜYEN bir açıklık vardır:

    B(t) = B * min(t / t_f, 1)

Gövde, ek bir residual terimiyle temsil edilir (sınır koşulu değil!):

    wall_res = exp(-((x-x_dam)/w)^2) * sigmoid((|y-y_c| - B(t)/2)/s) * u  ->  0

Yani "gövdenin durduğu yerde normal hız sıfır olsun". Bu terim B ve t_f'e
türevlenebilir biçimde bağlı olduğundan ağ, breach geometrisini bir PARAMETRE
olarak öğrenir. Breach hidrografı dışarıdan verilmez — akış fizikten doğar.

BOYUTSUZLAŞTIRMA (koşullanma için kritik)
-----------------------------------------
h0 aralığı 5–25 m, hız ölçeği sqrt(g*h0) ile değişir. Ağ ham metre tahmin ederse
kötü koşullanır. Bu yüzden çıktılar parametreye göre ölçeklenir:

    h = h_min + softplus(raw_h) * h0
    u = sqrt(g*h0) * raw_u,      v = sqrt(g*h0) * raw_v

Ağ böylece boyutsuz oranları öğrenir; ölçek fizikten gelir.

KOLON DÜZENİ
------------
DeepXDE'nin GeometryXTime'ı zamanı SONA koyar. Hypercube(x,y,B,t_f,h0,n) x Time:

    [x, y, B, t_f, h0, n, t]  ->  indeksler 0..6

Çalıştırma:
    DDE_BACKEND=pytorch python src/pinn/pinn_parametric_2d.py --smoke
    DDE_BACKEND=pytorch python src/pinn/pinn_parametric_2d.py

Çıktılar (results/2d/):
    parametric_2d_model-*.pt
    parametric_2d_prediction.npz
    parametric_2d_loss.png
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import deepxde as dde  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

# GPU'da ufak kazanç: sabit girdi boyutunda en hızlı cuDNN çekirdeğini seç.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from physics.saint_venant_2d import make_pde_2d  # noqa: E402

# Girdi kolon düzeni (GeometryXTime zamanı sona koyar)
COL = {"x": 0, "y": 1, "B": 2, "t_f": 3, "h0": 4, "n": 5, "t": 6}
INPUT_DIM = 7
PARAM_KEYS = ["B", "t_f", "h0", "n"]


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges_2d.yaml"), "r") as f:
        return yaml.safe_load(f)


def _bounds(cfg):
    """Hypercube alt/üst sınırları: [x, y, B, t_f, h0, n] (t ayrı)."""
    pc = cfg["parametric_2d"]
    d, pr = pc["domain"], pc["param_ranges"]
    lower = [d["x_min"], d["y_min"]] + [pr[k]["min"] for k in PARAM_KEYS]
    upper = [d["x_max"], d["y_max"]] + [pr[k]["max"] for k in PARAM_KEYS]
    return np.array(lower, dtype=float), np.array(upper, dtype=float), d["T"]


def breach_width_t(B, t, t_f):
    """B(t) = B * min(t/t_f, 1) — torch/numpy uyumlu."""
    if isinstance(t, torch.Tensor):
        return B * torch.clamp(t / t_f, 0.0, 1.0)
    return B * np.clip(t / t_f, 0.0, 1.0)


def make_parametric_pde(cfg):
    """2D Saint-Venant + zamana/parametreye bağlı gövde (wall) residual'ı."""
    pc = cfg["parametric_2d"]
    g = cfg["physics"]["g"]
    dam = pc["dam"]
    x_dam, y_c = dam["x_dam"], dam["y_c"]
    w_wall, s_open = dam["w_wall"], dam["s_open"]

    base = make_pde_2d(g=g, S0x=cfg["physics"]["S0x"], S0y=cfg["physics"]["S0y"],
                       nu=pc["nu"], n_col=COL["n"], t_col=COL["t"])

    def pde(X, Y):
        res = base(X, Y)                      # [continuity, x_mom, y_mom]

        x = X[:, COL["x"]:COL["x"] + 1]
        y = X[:, COL["y"]:COL["y"] + 1]
        B = X[:, COL["B"]:COL["B"] + 1]
        t_f = X[:, COL["t_f"]:COL["t_f"] + 1]
        t = X[:, COL["t"]:COL["t"] + 1]
        u = Y[:, 1:2]

        Bt = breach_width_t(B, t, t_f)
        # gövde çizgisine yakınlık (Gauss) x açıklık DIŞINDA olma (sigmoid)
        near_dam = torch.exp(-((x - x_dam) / w_wall) ** 2)
        outside = torch.sigmoid((torch.abs(y - y_c) - 0.5 * Bt) / s_open)
        res.append(near_dam * outside * u)     # gövdede normal hız -> 0
        return res

    return pde


def build_model(cfg: dict, train_cfg: dict) -> dde.Model:
    pc = cfg["parametric_2d"]
    g = cfg["physics"]["g"]
    d, dam = pc["domain"], pc["dam"]
    x_dam, eps_x = dam["x_dam"], dam["eps_x"]
    h_min = pc["h_min"]
    lower, upper, T = _bounds(cfg)
    x_max, y_min, y_max = d["x_max"], d["y_min"], d["y_max"]

    geom = dde.geometry.Hypercube(lower, upper)
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde = make_parametric_pde(cfg)

    # ---- Başlangıç koşulu: rezervuar (x<x_dam) h0, mansap h_min ----
    def ic_h_func(X):
        x = X[:, COL["x"]:COL["x"] + 1]
        h0 = X[:, COL["h0"]:COL["h0"] + 1]
        return h_min + (h0 - h_min) * 0.5 * (1.0 - np.tanh((x - x_dam) / eps_x))

    def zero_func(X):
        return np.zeros((X.shape[0], 1))

    ic_h = dde.icbc.IC(geomtime, ic_h_func, lambda _, on_i: on_i, component=0)
    ic_u = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=1)
    ic_v = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=2)

    # ---- Sınır koşulları ----
    # DİKKAT: Hypercube 6 boyutlu olduğu için `on_boundary` parametre yüzlerinde de
    # True döner. Bu yüzden her BC'de UZAYSAL koordinat açıkça kontrol edilir.
    def on_back_wall(X, on_boundary):        # x = 0 : rezervuar arka duvarı
        return on_boundary and np.isclose(X[COL["x"]], 0.0)

    def on_side_walls(X, on_boundary):       # y = y_min / y_max : yan duvarlar
        return on_boundary and (np.isclose(X[COL["y"]], y_min)
                                or np.isclose(X[COL["y"]], y_max))

    def on_far_field(X, on_boundary):        # x = x_max : dalga T içinde ulaşmaz
        return on_boundary and np.isclose(X[COL["x"]], x_max)

    bc_back_u = dde.icbc.DirichletBC(geomtime, zero_func, on_back_wall, component=1)
    bc_side_v = dde.icbc.DirichletBC(geomtime, zero_func, on_side_walls, component=2)
    bc_far_h = dde.icbc.DirichletBC(
        geomtime, lambda X: np.full((X.shape[0], 1), h_min), on_far_field, component=0)
    bc_far_u = dde.icbc.DirichletBC(geomtime, zero_func, on_far_field, component=1)

    data = dde.data.TimePDE(
        geomtime, pde,
        [ic_h, ic_u, ic_v, bc_back_u, bc_side_v, bc_far_h, bc_far_u],
        num_domain=train_cfg["num_domain"],
        num_boundary=train_cfg["num_boundary"],
        num_initial=train_cfg["num_initial"],
    )

    layers = pc["training"]["layers"]
    activation = pc["training"]["activation"]
    net = dde.nn.FNN([INPUT_DIM] + layers + [3], activation, "Glorot normal")

    lo = torch.tensor(lower, dtype=torch.float32)
    hi = torch.tensor(upper, dtype=torch.float32)

    def feature_transform(X):
        """Tüm girdileri [-1,1]'e getir (x,y,B,t_f,h0,n ve t)."""
        sp = 2.0 * (X[:, :6] - lo) / (hi - lo) - 1.0
        tn = 2.0 * X[:, COL["t"]:COL["t"] + 1] / T - 1.0
        return torch.cat([sp, tn], dim=1)

    net.apply_feature_transform(feature_transform)

    def output_transform(X, Y):
        """Parametreye göre boyutsuzlaştırma: ölçek fizikten, oran ağdan."""
        h0 = X[:, COL["h0"]:COL["h0"] + 1]
        c = torch.sqrt(g * h0)                       # hız ölçeği
        h = h_min + torch.nn.functional.softplus(Y[:, 0:1]) * h0
        u = c * Y[:, 1:2]
        v = c * Y[:, 2:3]
        return torch.cat([h, u, v], dim=1)

    net.apply_output_transform(output_transform)
    return dde.Model(data, net)


def _save_prediction(model, cfg, results_dir):
    """Birkaç temsili (B,t_f,h0,n) senaryosunda h(x,y) haritası kaydeder."""
    pc = cfg["parametric_2d"]
    d = pc["domain"]
    lower, upper, T = _bounds(cfg)
    pr = pc["param_ranges"]

    # Temsili senaryolar: dar/yavaş, orta, geniş/hızlı breach
    scenarios = [
        dict(name="dar-yavas",  B=pr["B"]["min"], t_f=pr["t_f"]["max"],
             h0=pr["h0"]["min"], n=0.04),
        dict(name="orta",       B=0.5 * (pr["B"]["min"] + pr["B"]["max"]),
             t_f=0.5 * (pr["t_f"]["min"] + pr["t_f"]["max"]),
             h0=0.5 * (pr["h0"]["min"] + pr["h0"]["max"]), n=0.04),
        dict(name="genis-hizli", B=pr["B"]["max"], t_f=pr["t_f"]["min"],
             h0=pr["h0"]["max"], n=0.04),
    ]
    times = np.array([0.25 * T, 0.5 * T, T], dtype=float)

    nx, ny = 160, 80
    xs = np.linspace(d["x_min"], d["x_max"], nx)
    ys = np.linspace(d["y_min"], d["y_max"], ny)
    XX, YY = np.meshgrid(xs, ys)

    H = np.zeros((len(scenarios), len(times), ny, nx))
    U = np.zeros_like(H)
    V = np.zeros_like(H)
    for si, sc in enumerate(scenarios):
        flat = np.zeros((XX.size, INPUT_DIM))
        flat[:, COL["x"]] = XX.ravel()
        flat[:, COL["y"]] = YY.ravel()
        flat[:, COL["B"]] = sc["B"]
        flat[:, COL["t_f"]] = sc["t_f"]
        flat[:, COL["h0"]] = sc["h0"]
        flat[:, COL["n"]] = sc["n"]
        for k, t in enumerate(times):
            flat[:, COL["t"]] = t
            pred = model.predict(flat)
            H[si, k] = pred[:, 0].reshape(ny, nx)
            U[si, k] = pred[:, 1].reshape(ny, nx)
            V[si, k] = pred[:, 2].reshape(ny, nx)

    out = os.path.join(results_dir, "parametric_2d_prediction.npz")
    np.savez(out, x=xs, y=ys, t=times, h=H, u=U, v=V,
             scenario_names=np.array([s["name"] for s in scenarios]),
             scenario_B=np.array([s["B"] for s in scenarios]),
             scenario_tf=np.array([s["t_f"] for s in scenarios]),
             scenario_h0=np.array([s["h0"] for s in scenarios]),
             scenario_n=np.array([s["n"] for s in scenarios]),
             x_dam=pc["dam"]["x_dam"], y_c=pc["dam"]["y_c"], h_min=pc["h_min"])
    return out


def train(smoke: bool = False) -> None:
    cfg = load_config()
    pc = cfg["parametric_2d"]
    train_cfg = pc["training"]["smoke" if smoke else "full"]
    results_dir = os.path.join(ROOT, "results", "2d")
    os.makedirs(results_dir, exist_ok=True)

    lower, upper, T = _bounds(cfg)
    print("[pinn_parametric_2d] Faz 2C — parametrik 2D")
    print(f"  Girdi ({INPUT_DIM}): {list(COL.keys())}")
    print(f"  Parametre aralıkları: " +
          ", ".join(f"{k}∈[{pc['param_ranges'][k]['min']:g},"
                    f"{pc['param_ranges'][k]['max']:g}]" for k in PARAM_KEYS))
    print(f"  Alan: {upper[0]:g}×{upper[1]:g} m, T={T:g} s, x_dam={pc['dam']['x_dam']:g} m")

    model = build_model(cfg, train_cfg)

    # Loss sırası: pde(3) + wall(1) + ic(3) + bc(4) = 11
    loss_weights = [1.0, 1.0, 1.0, 10.0,      # pde + gövde
                    10.0, 10.0, 10.0,          # ic h,u,v
                    5.0, 5.0, 5.0, 5.0]        # bc
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    print(f"[pinn_parametric_2d] Adam ({train_cfg['adam_iters']} iter)...")
    losshistory, _ = model.train(iterations=train_cfg["adam_iters"], display_every=500)

    if train_cfg["lbfgs"]:
        print("[pinn_parametric_2d] L-BFGS ince ayar (maxiter=3000)...")
        dde.optimizers.set_LBFGS_options(maxiter=3000)  # varsayılan 15000 çok uzun
        model.compile("L-BFGS")
        losshistory, _ = model.train()

    model.save(os.path.join(results_dir, "parametric_2d_model"))
    out = _save_prediction(model, cfg, results_dir)

    try:
        dde.utils.plot_loss_history(losshistory)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(results_dir, "parametric_2d_loss.png"), dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"[pinn_parametric_2d] loss grafiği atlandı: {exc}")

    print(f"[pinn_parametric_2d] ✓ Tamamlandı. Tahmin: {out}")
    if smoke:
        print("[pinn_parametric_2d] (smoke — convergence BEKLENMEZ; 7 boyutlu girdi "
              "uzayı için tam eğitim GPU gerektirir.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz 2C parametrik 2D dam-break PINN")
    parser.add_argument("--smoke", action="store_true",
                        help="Hızlı uçtan uca test (az iterasyon, L-BFGS yok)")
    args = parser.parse_args()
    train(smoke=args.smoke)
