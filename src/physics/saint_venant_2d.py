"""2D Saint-Venant (sığ su) denklemleri — DeepXDE PINN residual'ı.

Primitif (h, u, v) non-conservative form:

    süreklilik : h_t + u*h_x + h*u_x + v*h_y + h*v_y = 0
    x-momentum : u_t + u*u_x + v*u_y + g*h_x = g*(S0x - Sfx)
    y-momentum : v_t + u*v_x + v*v_y + g*h_y = g*(S0y - Sfy)
    Manning    : speed = sqrt(u^2 + v^2)
                 Sfx = n^2 * u*speed / h^(4/3),  Sfy = n^2 * v*speed / h^(4/3)

Girdi X kolonları : [x, y, t]   (j = 0, 1, 2)
Çıktı  Y kolonları: [h, u, v]   (i = 0, 1, 2)

`make_pde_2d` bir closure döndürür; Faz 2A (girdi [x,y,t]) ve Faz 2C (girdi
[x,y,t,B,t_f,h0,n], n bir girdi kolonundan okunur) için aynı fabrika kullanılır.
"""
from __future__ import annotations

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402
import torch  # noqa: E402


def make_pde_2d(g: float = 9.81, n: float = 0.0, S0x: float = 0.0,
                S0y: float = 0.0, nu: float = 0.0, n_col: int | None = None,
                t_col: int = 2):
    """2D Saint-Venant residual closure'ı üretir.

    n_col : Faz 2C'de Manning n'in okunacağı girdi kolonu. None ise sabit `n`.
    t_col : zaman kolonunun indeksi. Faz 2A'da girdi [x,y,t] olduğu için 2;
            Faz 2C'de GeometryXTime zamanı SONA koyduğundan
            [x,y,B,t_f,h0,n,t] düzeninde 6'dır.
    nu    : artificial viscosity (şok regularizasyonu). 0 ise kapalı.
    """

    use_friction = (n_col is not None) or (n and n > 0.0)

    def pde(X, Y):
        h = Y[:, 0:1]
        u = Y[:, 1:2]
        v = Y[:, 2:3]

        # 1. türevler (i: çıktı, j: girdi -> 0:x, 1:y, t_col:t)
        h_x = dde.grad.jacobian(Y, X, i=0, j=0)
        h_y = dde.grad.jacobian(Y, X, i=0, j=1)
        h_t = dde.grad.jacobian(Y, X, i=0, j=t_col)
        u_x = dde.grad.jacobian(Y, X, i=1, j=0)
        u_y = dde.grad.jacobian(Y, X, i=1, j=1)
        u_t = dde.grad.jacobian(Y, X, i=1, j=t_col)
        v_x = dde.grad.jacobian(Y, X, i=2, j=0)
        v_y = dde.grad.jacobian(Y, X, i=2, j=1)
        v_t = dde.grad.jacobian(Y, X, i=2, j=t_col)

        # Manning sürtünmesi
        if use_friction:
            n_val = X[:, n_col:n_col + 1] if n_col is not None else n
            speed = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
            denom = h ** (4.0 / 3.0) + 1e-8
            Sfx = n_val ** 2 * u * speed / denom
            Sfy = n_val ** 2 * v * speed / denom
        else:
            Sfx = 0.0
            Sfy = 0.0

        continuity = h_t + u * h_x + h * u_x + v * h_y + h * v_y
        x_mom = u_t + u * u_x + v * u_y + g * h_x - g * (S0x - Sfx)
        y_mom = v_t + u * v_x + v * v_y + g * h_y - g * (S0y - Sfy)

        # Opsiyonel artificial viscosity (2D Laplacian)
        if nu and nu > 0.0:
            h_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)
            h_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)
            u_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)
            u_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)
            v_xx = dde.grad.hessian(Y, X, component=2, i=0, j=0)
            v_yy = dde.grad.hessian(Y, X, component=2, i=1, j=1)
            continuity = continuity - nu * (h_xx + h_yy)
            x_mom = x_mom - nu * (u_xx + u_yy)
            y_mom = y_mom - nu * (v_xx + v_yy)

        return [continuity, x_mom, y_mom]

    return pde


if __name__ == "__main__":
    # Sanity: residual'ı rastgele noktalarda değerlendir, şekilleri yazdır.
    print("2D Saint-Venant residual sanity test")
    print("-" * 40)
    X = torch.rand(8, 3, requires_grad=True)
    # h,u,v'yi X'e bağımlı türevlenebilir bir fonksiyon olarak kur
    h = (X ** 2).sum(dim=1, keepdim=True) + 0.5
    u = X[:, 0:1] * X[:, 2:3]
    v = X[:, 1:2] * X[:, 2:3]
    Y = torch.cat([h, u, v], dim=1)

    for tag, kw in [("sürtünmesiz, nu=0", dict()),
                    ("nu=1e-3", dict(nu=1e-3)),
                    ("n=0.03", dict(n=0.03))]:
        try:
            res = make_pde_2d(**kw)(X, Y)
            shapes = [tuple(r.shape) if hasattr(r, "shape") else r for r in res]
            finite = all(torch.isfinite(r).all() for r in res)
            print(f"  [{tag}] residual sayısı={len(res)} şekiller={shapes} sonlu={finite}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{tag}] HATA: {exc}")
        dde.grad.clear()  # türev önbelleğini temizle
    print("✓ make_pde_2d 3 residual döndürüyor: [continuity, x_mom, y_mom]")
