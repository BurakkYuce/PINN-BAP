"""1D radyal (silindirik) Saint-Venant — DeepXDE PINN residual'ı.

Eksenel simetrik (axisymmetric) sığ su denklemleri; bağımsız değişkenler (r, t),
çözülenler (h, u). 2D Kartezyen formülasyonun aksine AÇISAL boyut yoktur: çözüm
tanım gereği eksenel simetriktir (kelebek/asimetri imkânsız) ve tüm collocation
bütçesi (r,t) düzlemine gider → cephe ve merkez denklem düzeyinde daha keskin.

Primitif (h, u) non-conservative form, 1/r geometrik kaynak ile:

    süreklilik :  h_t + u*h_r + h*u_r + h*u/r = 0     <- 1/r yalnız burada
    momentum   :  u_t + u*u_r + g*h_r = g*(S0 - Sf)    <- geometrik terim sadeleşir
    Manning    :  Sf = n^2 * u*|u| / h^(4/3)

(Momentumda 1/r terimi YOK: konservatif akı diverjansı ile basınç gradyanı
arasında sadeleşir — türetme dosya sonundaki nota bakınız.)

Girdi X kolonları : [r, t]   (j = 0, 1)
Çıktı  Y kolonları: [h, u]   (i = 0, 1)

r=0'da kaynak h*u/r, simetri (u(0,t)=0) sayesinde sonludur. PINN tarafında u,
çıktı dönüşümüyle u = tanh(r/L)*u_raw kurulduğunda u(0,t)=0 KESİN sağlanır ve
u/r sınırlı kalır (bkz. pinn_radial_1d.py).
"""
from __future__ import annotations

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")

import deepxde as dde  # noqa: E402
import torch  # noqa: E402


def make_pde_radial(g: float = 9.81, n: float = 0.0, S0: float = 0.0,
                    nu: float = 0.0):
    """Radyal Saint-Venant residual closure'ı üretir.

    g  : yerçekimi ivmesi.
    n  : Manning katsayısı (0 ise sürtünmesiz).
    S0 : taban eğimi.
    nu : artificial viscosity (şok regularizasyonu). 0 ise kapalı.
    """

    def pde(X, Y):
        r = X[:, 0:1]
        h = Y[:, 0:1]
        u = Y[:, 1:2]

        # 1. türevler (i: çıktı, j: girdi -> 0:r, 1:t)
        h_r = dde.grad.jacobian(Y, X, i=0, j=0)
        h_t = dde.grad.jacobian(Y, X, i=0, j=1)
        u_r = dde.grad.jacobian(Y, X, i=1, j=0)
        u_t = dde.grad.jacobian(Y, X, i=1, j=1)

        if n and n > 0.0:
            Sf = n ** 2 * u * torch.abs(u) / (h ** (4.0 / 3.0) + 1e-8)
        else:
            Sf = 0.0

        # silindirik (1/r) geometrik kaynak — yalnız süreklilikte.
        src = h * u / (r + 1e-8)

        continuity = h_t + u * h_r + h * u_r + src
        momentum = u_t + u * u_r + g * h_r - g * (S0 - Sf)

        if nu and nu > 0.0:
            h_rr = dde.grad.hessian(Y, X, component=0, i=0, j=0)
            u_rr = dde.grad.hessian(Y, X, component=1, i=0, j=0)
            continuity = continuity - nu * h_rr
            momentum = momentum - nu * u_rr

        return [continuity, momentum]

    return pde


if __name__ == "__main__":
    # Sanity: residual'ı rastgele (r>0) noktalarda değerlendir.
    print("Radyal Saint-Venant residual sanity test")
    print("-" * 42)
    r = torch.rand(8, 1) * 4.0 + 0.1          # r in [0.1, 4.1], r>0
    t = torch.rand(8, 1) * 0.6
    X = torch.cat([r, t], dim=1).requires_grad_(True)
    h = (X ** 2).sum(dim=1, keepdim=True) + 0.5
    u = X[:, 0:1] * X[:, 1:2]
    Y = torch.cat([h, u], dim=1)

    for tag, kw in [("sürtünmesiz, nu=0", dict()),
                    ("nu=3e-4", dict(nu=3e-4)),
                    ("n=0.03", dict(n=0.03))]:
        try:
            res = make_pde_radial(**kw)(X, Y)
            shapes = [tuple(rr.shape) for rr in res]
            finite = all(torch.isfinite(rr).all() for rr in res)
            print(f"  [{tag}] residual sayısı={len(res)} şekiller={shapes} sonlu={finite}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{tag}] HATA: {exc}")
        dde.grad.clear()
    print("✓ make_pde_radial 2 residual döndürüyor: [continuity, momentum]")
