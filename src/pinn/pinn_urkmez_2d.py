"""Faz 2B — Ürkmez Barajı 2D taşkın yayılımı PINN'i.

Gerçek vaka: Ürkmez Barajı (toprak dolgu) kırılması sonrası mansap taşkın ovasında
2D Saint-Venant çözümü. Referans benchmark: FLO-2D (Haltaş, Tayfur & Elçi 2016,
~%14 derinlik hatası) — karşılaştırma validation/compare_urkmez_flo2d.py ile.

KURULUM
-------
Alan, barajın MANSABINDAKİ taşkın ovasıdır (4000 × 3000 m). Breach, x = 0
kenarında y_c merkezli B genişliğinde bir açıklıktır ve oraya bir HİDROGRAF
beslenir:

    Üçgen breach hidrografı:  Q(t): 0 -> Q_peak (t = t_f) -> 0 (t = t_f(1+rf))
    Birim debi:               q(t) = Q(t) / B
    Kritik akış koşulu:       h_in = (q²/g)^(1/3),   u_in = q / h_in

Kritik akış, breach ağzındaki klasik kabuldür (Froude = 1) ve derinlik/hızı
debiden tutarlı biçimde türetir.

⚠ TOPOGRAFYA: Taban DÜZ + sabit eğim (S0x) kabul edilmiştir. Gerçek DEM
(cases/urkmez.json → domain_2d yer tutucudur) eklenmeden FLO-2D ile hücre-hücre
karşılaştırma NİTELİKSELDİR. Bu, bilinçli bir sınırlamadır.

⚠ ZAMAN PENCERESİ: t_f = 2052 s (0.57 saat) olduğundan varsayılan T = 900 s
yalnızca hidrografın YÜKSELEN KOLUNU kapsar. Pik ve resesyon için config'de
T ≈ 2500 s yapılmalı (çok daha uzun eğitim gerekir).

Çalıştırma:
    DDE_BACKEND=pytorch python src/pinn/pinn_urkmez_2d.py --smoke
    DDE_BACKEND=pytorch python src/pinn/pinn_urkmez_2d.py

Çıktılar (results/2d/):
    urkmez_2d_model-*.pt, urkmez_2d_prediction.npz, urkmez_2d_loss.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import deepxde as dde  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from physics.saint_venant_2d import make_pde_2d  # noqa: E402

G = 9.81


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges_2d.yaml"), "r") as f:
        return yaml.safe_load(f)


def load_case() -> dict:
    with open(os.path.join(ROOT, "cases", "urkmez.json"), "r") as f:
        return json.load(f)


def breach_hydrograph(t, Q_peak: float, t_f: float, recession_factor: float = 3.0):
    """Üçgen breach hidrografı Q(t) [m³/s]. t: skaler veya numpy dizisi."""
    t = np.asarray(t, dtype=float)
    t_rec = recession_factor * t_f
    rising = Q_peak * (t / t_f)
    receding = Q_peak * (1.0 - (t - t_f) / t_rec)
    Q = np.where(t < t_f, rising, receding)
    return np.clip(Q, 0.0, None)


def inflow_state(t, Q_peak, t_f, B, recession_factor=3.0, h_floor=1e-3):
    """Breach ağzında kritik akış: (h_in, u_in) [m, m/s]."""
    Q = breach_hydrograph(t, Q_peak, t_f, recession_factor)
    q = Q / B                                    # birim debi (m²/s)
    h_in = np.maximum((q ** 2 / G) ** (1.0 / 3.0), h_floor)
    u_in = q / h_in
    return h_in, u_in


def build_model(cfg: dict, train_cfg: dict) -> dde.Model:
    uc = cfg["urkmez_2d"]
    d, br, bed = uc["domain"], uc["breach"], uc["bed"]
    x_min, x_max = d["x_min"], d["x_max"]
    y_min, y_max = d["y_min"], d["y_max"]
    T = d["T"]
    y_c, B, t_f = br["y_c"], br["B"], br["t_f"]
    Q_peak, rf = br["Q_peak"], br["recession_factor"]
    h_min, n_manning, nu = uc["h_min"], uc["manning_n"], uc["nu"]

    geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde = make_pde_2d(g=G, n=n_manning, S0x=bed["S0x"], S0y=bed["S0y"], nu=nu)

    def zero_func(X):
        return np.zeros((X.shape[0], 1))

    # ---- Başlangıç: kuru-ıslak taşkın ovası (h_min), durgun ----
    ic_h = dde.icbc.IC(geomtime, lambda X: np.full((X.shape[0], 1), h_min),
                       lambda _, on_i: on_i, component=0)
    ic_u = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=1)
    ic_v = dde.icbc.IC(geomtime, zero_func, lambda _, on_i: on_i, component=2)

    # ---- Breach girişi: PointSetBC ----
    # Breach (63.7 m) alanın y-genişliğinin (~3000 m) %2'sinden küçük olduğundan,
    # rastgele sınır örneklemesi burayı ıskalar. Bu yüzden noktalar ELLE konur.
    n_y, n_t = 40, 120
    yb = np.linspace(y_c - B / 2, y_c + B / 2, n_y)
    tb = np.linspace(0.0, T, n_t)
    YB, TB = np.meshgrid(yb, tb)
    pts_in = np.column_stack([np.zeros(YB.size), YB.ravel(), TB.ravel()])
    h_in, u_in = inflow_state(pts_in[:, 2], Q_peak, t_f, B, rf, h_floor=h_min)
    bc_in_h = dde.icbc.PointSetBC(pts_in, h_in.reshape(-1, 1), component=0)
    bc_in_u = dde.icbc.PointSetBC(pts_in, u_in.reshape(-1, 1), component=1)

    # ---- x=0'da breach DIŞI: duvar (u=0) ----
    def on_inlet_wall(X, on_boundary):
        return (on_boundary and np.isclose(X[0], x_min)
                and abs(X[1] - y_c) > B / 2)

    bc_wall_u = dde.icbc.DirichletBC(geomtime, zero_func, on_inlet_wall, component=1)

    # ---- Yan duvarlar (y=y_min,y_max): v=0 ----
    def on_side(X, on_boundary):
        return on_boundary and (np.isclose(X[1], y_min) or np.isclose(X[1], y_max))

    bc_side_v = dde.icbc.DirichletBC(geomtime, zero_func, on_side, component=2)

    # ---- Çıkış (x=x_max): transmissive (sıfır gradyan) ----
    # Dalga T içinde alanı geçtiğinden Dirichlet YANLIŞ olur; Neumann kullanılır.
    def on_outlet(X, on_boundary):
        return on_boundary and np.isclose(X[0], x_max)

    bc_out_h = dde.icbc.NeumannBC(geomtime, zero_func, on_outlet, component=0)
    bc_out_u = dde.icbc.NeumannBC(geomtime, zero_func, on_outlet, component=1)

    data = dde.data.TimePDE(
        geomtime, pde,
        [ic_h, ic_u, ic_v, bc_in_h, bc_in_u, bc_wall_u, bc_side_v, bc_out_h, bc_out_u],
        num_domain=train_cfg["num_domain"],
        num_boundary=train_cfg["num_boundary"],
        num_initial=train_cfg["num_initial"],
    )

    layers = uc["training"]["layers"]
    net = dde.nn.FNN([3] + layers + [3], uc["training"]["activation"], "Glorot normal")

    # Ölçekler: pik kritik derinlik ve hız mertebesi
    h_scale = float(max(inflow_state(np.array([t_f]), Q_peak, t_f, B, rf)[0][0], 1.0))
    u_scale = float(max(inflow_state(np.array([t_f]), Q_peak, t_f, B, rf)[1][0], 1.0))

    def feature_transform(X):
        xs = 2.0 * (X[:, 0:1] - x_min) / (x_max - x_min) - 1.0
        ys = 2.0 * (X[:, 1:2] - y_min) / (y_max - y_min) - 1.0
        ts = 2.0 * X[:, 2:3] / T - 1.0
        return torch.cat([xs, ys, ts], dim=1)

    net.apply_feature_transform(feature_transform)

    def output_transform(X, Y):
        h = h_min + torch.nn.functional.softplus(Y[:, 0:1]) * h_scale
        return torch.cat([h, u_scale * Y[:, 1:2], u_scale * Y[:, 2:3]], dim=1)

    net.apply_output_transform(output_transform)
    return dde.Model(data, net), (h_scale, u_scale)


def _save_prediction(model, cfg, results_dir):
    uc = cfg["urkmez_2d"]
    d = uc["domain"]
    times = np.array(uc["training"]["predict_times"], dtype=float)
    nx, ny = 200, 150
    xs = np.linspace(d["x_min"], d["x_max"], nx)
    ys = np.linspace(d["y_min"], d["y_max"], ny)
    XX, YY = np.meshgrid(xs, ys)

    H = np.zeros((len(times), ny, nx)); U = np.zeros_like(H); V = np.zeros_like(H)
    flat = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])
    for k, t in enumerate(times):
        flat[:, 2] = t
        p = model.predict(flat)
        H[k] = p[:, 0].reshape(ny, nx)
        U[k] = p[:, 1].reshape(ny, nx)
        V[k] = p[:, 2].reshape(ny, nx)

    # FLO-2D karşılaştırması maksimum derinlik rasterı üzerinden yapılır
    h_max = H.max(axis=0)

    out = os.path.join(results_dir, "urkmez_2d_prediction.npz")
    np.savez(out, x=xs, y=ys, t=times, h=H, u=U, v=V, h_max=h_max,
             y_c=uc["breach"]["y_c"], B=uc["breach"]["B"], h_min=uc["h_min"])
    return out


def train(smoke: bool = False) -> None:
    cfg = load_config()
    case = load_case()
    uc = cfg["urkmez_2d"]
    train_cfg = uc["training"]["smoke" if smoke else "full"]
    results_dir = os.path.join(ROOT, "results", "2d")
    os.makedirs(results_dir, exist_ok=True)

    br = uc["breach"]
    Qp, t_f, B = br["Q_peak"], br["t_f"], br["B"]
    h_pk, u_pk = inflow_state(np.array([t_f]), Qp, t_f, B, br["recession_factor"])
    print(f"[pinn_urkmez_2d] Vaka: {case['name']} ({case['dam_type_tr']})")
    print(f"  Breach: B={B} m, t_f={t_f:.0f} s, Q_peak={Qp} m³/s")
    print(f"  Pik girişte kritik akış: h={h_pk[0]:.2f} m, u={u_pk[0]:.2f} m/s")
    print(f"  Alan: {uc['domain']['x_max']:.0f}×{uc['domain']['y_max']:.0f} m, "
          f"T={uc['domain']['T']:.0f} s")
    if uc["domain"]["T"] < t_f:
        print(f"  ⚠ T < t_f: yalnızca hidrografın YÜKSELEN kolu simüle ediliyor.")

    model, _ = build_model(cfg, train_cfg)

    # pde(3) + ic(3) + bc_in(2) + wall + side + out(2) = 12
    loss_weights = [1., 1., 1.,  10., 10., 10.,  20., 20.,  5., 5.,  1., 1.]
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    print(f"[pinn_urkmez_2d] Adam ({train_cfg['adam_iters']} iter)...")
    losshistory, _ = model.train(iterations=train_cfg["adam_iters"], display_every=500)

    if train_cfg["lbfgs"]:
        print("[pinn_urkmez_2d] L-BFGS ince ayar...")
        model.compile("L-BFGS")
        losshistory, _ = model.train()

    model.save(os.path.join(results_dir, "urkmez_2d_model"))
    out = _save_prediction(model, cfg, results_dir)

    try:
        dde.utils.plot_loss_history(losshistory)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.savefig(os.path.join(results_dir, "urkmez_2d_loss.png"), dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        print(f"[pinn_urkmez_2d] loss grafiği atlandı: {exc}")

    print(f"[pinn_urkmez_2d] ✓ Tamamlandı. Tahmin: {out}")
    print("  Sonraki: python validation/compare_urkmez_flo2d.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz 2B Ürkmez 2D taşkın PINN'i")
    parser.add_argument("--smoke", action="store_true",
                        help="Hızlı uçtan uca test (az iterasyon, L-BFGS yok)")
    args = parser.parse_args()
    train(smoke=args.smoke)
