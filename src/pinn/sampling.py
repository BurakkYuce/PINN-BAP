"""Parametre uzayından örnekleme — Latin Hypercube Sampling (LHS).

Faz 3'te ağ (B, t_f, h0, n) parametre ailesini öğrenir. Eğitim sırasında bu uzaydan
düzgün dağılımlı örnekler almak için LHS kullanılır (rastgele örneklemeye göre daha
iyi uzay kapsaması).

Çalıştırma:
    python src/pinn/sampling.py        # config aralıklarından örnek tablosu yazdırır
"""
from __future__ import annotations

import os

import numpy as np
import yaml
from scipy.stats import qmc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_param_ranges() -> dict:
    with open(os.path.join(ROOT, "config", "param_ranges.yaml"), "r") as f:
        return yaml.safe_load(f)["param_ranges"]


def sample_params(n_samples: int, ranges: dict | None = None, seed: int = 0):
    """Parametre uzayından LHS ile `n_samples` örnek üretir.

    Döndürür
    --------
    keys : list[str]      parametre adları (kolon sırası)
    samples : np.ndarray  (n_samples, n_params) ölçeklenmiş örnekler
    """
    if ranges is None:
        ranges = load_param_ranges()

    keys = list(ranges.keys())
    lower = np.array([ranges[k]["min"] for k in keys], dtype=float)
    upper = np.array([ranges[k]["max"] for k in keys], dtype=float)

    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit = sampler.random(n_samples)          # [0,1]^d
    samples = qmc.scale(unit, lower, upper)    # fiziksel aralıklara ölçekle
    return keys, samples


if __name__ == "__main__":
    keys, samples = sample_params(8, seed=42)
    print("Latin Hypercube örnekleri (Faz 3 parametre uzayı)")
    print("-" * 52)
    print("  " + "  ".join(f"{k:>8}" for k in keys))
    for row in samples:
        print("  " + "  ".join(f"{v:8.3f}" for v in row))
