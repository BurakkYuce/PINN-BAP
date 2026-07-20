"""1D Saint-Venant (sığ su) denklemleri — DeepXDE PINN residual'ı.

Primitif (h, u) değişkenleriyle non-conservative yazım; otomatik türev (autodiff)
için temiz form:

    süreklilik :  h_t + u*h_x + h*u_x = 0
    momentum   :  u_t + u*u_x + g*h_x = g*(S0 - Sf)
    Manning    :  Sf = n^2 * u*|u| / h^(4/3)

`make_pde` bir closure (pde fonksiyonu) döndürür. Aynı fabrika hem Faz 1 (girdi
[x, t]) hem Faz 3 (girdi [x, t, B, t_f, h0, n]) için kullanılır: Faz 3'te Manning n
sabit değil, ağ girdisinin bir kolonundan (`n_col`) okunur.
"""
from __future__ import annotations

import os

# DeepXDE backend'i, deepxde import edilmeden ÖNCE sabitlenmeli.
os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402
import torch  # noqa: E402  (PyTorch backend; u*|u| ve pozitiflik için)


def make_pde(g: float = 9.81, n: float = 0.0, S0: float = 0.0,
             nu: float = 0.0, n_col: int | None = None):
    """Saint-Venant residual closure'ı üretir.

    Parametreler
    ------------
    g : float
        Yerçekimi ivmesi.
    n : float
        Manning katsayısı (sabit senaryo; Faz 1 için tipik olarak 0).
    S0 : float
        Taban eğimi.
    nu : float
        Artificial viscosity katsayısı (şok regularizasyonu). 0 ise kapalı.
        Dam-break süreksizliğinde PINN convergence'ı için küçük bir nu (ör. 1e-3)
        yardımcı olabilir.
    n_col : int | None
        Faz 3 için Manning n'in okunacağı girdi kolonu indeksi (ör. 5).
        None ise sabit `n` kullanılır.
    """

    def pde(x, y):
        h = y[:, 0:1]
        u = y[:, 1:2]

        # Birinci türevler: i = çıktı bileşeni, j = girdi bileşeni (0:x, 1:t)
        h_x = dde.grad.jacobian(y, x, i=0, j=0)
        h_t = dde.grad.jacobian(y, x, i=0, j=1)
        u_x = dde.grad.jacobian(y, x, i=1, j=0)
        u_t = dde.grad.jacobian(y, x, i=1, j=1)

        # Manning sürtünme eğimi Sf = n^2 * u|u| / h^(4/3)
        if n_col is not None:
            n_val = x[:, n_col:n_col + 1]
            Sf = n_val ** 2 * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        elif n and n > 0.0:
            Sf = n ** 2 * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        else:
            Sf = 0.0

        continuity = h_t + u * h_x + h * u_x
        momentum = u_t + u * u_x + g * h_x - g * (S0 - Sf)

        # Opsiyonel artificial viscosity (şok yumuşatma)
        if nu and nu > 0.0:
            h_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            u_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
            continuity = continuity - nu * h_xx
            momentum = momentum - nu * u_xx

        return [continuity, momentum]

    return pde
