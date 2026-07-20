"""Faz 3 — Parametrik dam-break PINN (İSKELET).

Asıl katkı: ağ tek bir senaryoyu değil, bir senaryo AİLESİNİ öğrenir.

    h, u = f(x, t, B, t_f, h0, n)        girdi boyutu = 6

Bir kez eğitilince, parametre uzayındaki herhangi bir (B, t_f, h0, n) için yeniden
eğitmeden anında çözüm verir. Bu dosya kurulumu (ağ + residual + örnekleme) hazır
bir İSKELETtir; başlangıç/sınır koşullarının parametreye bağlı kısımları TODO ile
işaretlenmiştir ve Faz 1 doğrulandıktan SONRA doldurulacaktır.

Kademeli geliştirme önerisi (README'deki strateji):
    Faz 2: önce yalnızca h0'ı parametre yap (girdi 3D) — parametrik mantığı küçük
           ölçekte kanıtla; sonra B, t_f, n eklenir.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("DDE_BACKEND", "pytorch")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import deepxde as dde  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from physics.saint_venant import make_pde  # noqa: E402
from pinn.sampling import sample_params  # noqa: E402

# Girdi kolon düzeni:  [x, t, B, t_f, h0, n]
COL = {"x": 0, "t": 1, "B": 2, "t_f": 3, "h0": 4, "n": 5}
INPUT_DIM = 6


def load_config() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges.yaml"), "r") as f:
        return yaml.safe_load(f)


def build_network(cfg: dict) -> dde.nn.FNN:
    """6 girdili PINN ağı; h>0 için softplus output transform."""
    layers = cfg["training"]["layers"]
    activation = cfg["training"]["activation"]
    # Parametrik uzay daha geniş olduğundan baseline'a göre biraz daha derin/genis tutulabilir.
    net = dde.nn.FNN([INPUT_DIM] + layers + [16] + [2], activation, "Glorot normal")

    def output_transform(x, y):
        h = torch.nn.functional.softplus(y[:, 0:1])
        u = y[:, 1:2]
        return torch.cat([h, u], dim=1)

    net.apply_output_transform(output_transform)
    return net


def build_parametric_data(cfg: dict):
    """Parametrik collocation verisini kurar (İSKELET).

    Yaklaşım: (x, t) uzay-zaman noktaları DeepXDE geometrisinden, parametreler
    (B, t_f, h0, n) ise sampling.sample_params (LHS) ile üretilip her noktaya
    eklenir. Aşağıdaki residual ve örnekleme hazır; IC/BC kısmı TODO.
    """
    g = cfg["physics"]["g"]
    S0 = cfg["physics"]["S0"]
    nu = cfg["physics"]["nu"]

    # Manning n artık SABİT değil — girdinin 5. kolonundan okunur.
    pde = make_pde(g=g, S0=S0, nu=nu, n_col=COL["n"])  # noqa: F841  (data kurulumunda kullanılacak)

    # LHS ile parametre örnekleri (eğitimde anchor/collocation parametreleri).
    keys, param_samples = sample_params(n_samples=64)  # noqa: F841

    # TODO (Faz 3):
    #   1) Uzay-zaman geometrisi: dde.geometry.Interval x TimeDomain (x_max parametreye
    #      bağlanabilir) -> GeometryXTime.
    #   2) PointSetBC / custom sampler ile her collocation noktasına bir (B,t_f,h0,n)
    #      ekle; veya 4D Hypercube(B,t_f,h0,n) x (Interval x Time) birleşik geometri kur.
    #   3) IC: h(x,0) rezervuar seviyesi h0'a bağlı (membran solunda ~h0).
    #   4) Mansap giriş koşulu: kırılma hidrografı B ve t_f'e bağlı (Froehlich + pik
    #      debi); breach gelişimini zaman-bağımlı BC olarak uygula.
    #   5) dde.data.TimePDE(...) ile data nesnesini oluştur ve döndür.
    raise NotImplementedError(
        "Faz 3 parametrik veri kurulumu — IC/BC parametre bağımlılığı doldurulacak "
        "(Faz 1 Stoker doğrulamasından sonra)."
    )


def predict(model: dde.Model, x, t, B, t_f, h0, n):
    """Eğitilmiş parametrik model için tek (B,t_f,h0,n) senaryosunda tahmin.

    x : konum dizisi, t : zaman (skaler). Diğer parametreler skalerdir ve tüm
    noktalara yayılır. (Model eğitildikten sonra kullanılır.)
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n_pts = x.shape[0]
    cols = np.column_stack([
        x.ravel(),
        np.full(n_pts, t),
        np.full(n_pts, B),
        np.full(n_pts, t_f),
        np.full(n_pts, h0),
        np.full(n_pts, n),
    ])
    pred = model.predict(cols)
    return pred[:, 0], pred[:, 1]


def main() -> None:
    cfg = load_config()
    net = build_network(cfg)
    print("[pinn_parametric] İSKELET — Faz 3")
    print(f"  Girdi boyutu: {INPUT_DIM}  kolonlar: {list(COL.keys())}")
    print(f"  Ağ katmanları: {[INPUT_DIM] + cfg['training']['layers'] + [16, 2]}")
    print("  Residual: saint_venant.make_pde(n_col=5) — Manning n girdiden okunur.")
    print("  Örnekleme: sampling.sample_params (Latin Hypercube).")
    print("  Sonraki adım: build_parametric_data() içindeki IC/BC TODO'larını doldur.")


if __name__ == "__main__":
    main()
