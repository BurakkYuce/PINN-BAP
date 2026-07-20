# data/ — Doğrulama Verileri

Bu proje **eğitim için veri kullanmaz** (PINN denklemden öğrenir). Buradaki veriler
yalnızca **doğrulama / kanıt (validation)** içindir — modelin gerçek ölçümlere ne
kadar yakın olduğunu göstermek için.

```
data/
├── raw/         indirilen ham dosyalar (değiştirilmez)
└── processed/   ayrıştırılmış temiz CSV'ler (script üretir)
```

## Vosoughi vd. (2021) — Deneysel dam-break veri seti (KATMAN 2)

Açık erişimli laboratuvar dam-break ölçümleri. Bizim 1D Saint-Venant PINN'imiz için
ideal: temiz su (sedimentsiz) senaryoları + hem kuru hem ıslak mansap.

| Özellik | Değer |
|---|---|
| Rezervuar yüksekliği | H = 30 cm |
| Ölçüm konumu | 20 nokta (L = 0…552 cm) |
| Zaman anı | 15 (0.04…6.0 s) |
| Temiz-su senaryoları | 4 (kuru, ıslak 2/4/5 cm) |
| Tahmini kapı konumu | x0 ≈ 156–158 cm (analitik oturtmayla) |

**Temiz-su senaryoları (PINN doğrulaması için):**
- `vosoughi2021_clearwater_dry.csv` — kuru mansap → **Ritter** analitik
- `vosoughi2021_clearwater_wet2cm.csv` — ıslak 2 cm → **Stoker** analitik
- `vosoughi2021_clearwater_wet4cm.csv` — ıslak 4 cm → Stoker
- `vosoughi2021_clearwater_wet5cm.csv` — ıslak 5 cm → Stoker
- `vosoughi2021_clearwater.csv` — dördü birleşik (uzun-format)

CSV kolonları: `scenario, downstream, h_down_cm, L_cm, time_s, level_cm`

### Yeniden üretme

```bash
python src/data/parse_vosoughi2021.py     # ham yoksa indirir, CSV üretir
python validation/compare_experimental.py # deney vs analitik grafiği + RMSE
```

### Kaynak ve lisans

- **Atıf:** Vosoughi, F., Rakhshandehroo, G., Nikoo, M.R. (2021). *Experimental
  dataset on water levels in studying the influences of dry- and wet-bed downstream
  conditions...* Data in Brief.
- **Depo:** Mendeley Data, v3 — DOI [10.17632/nc573y67tp.3](https://data.mendeley.com/datasets/nc573y67tp/3)
- **Lisans:** CC BY 4.0 (Mendeley Data standardı). Kullanırken atıf verin.

> Not: İkinci bir DOI (`10.17632/zm7rr9ngn5.3`) %50–80 siltasyon senaryolarını
> içerir; bu projede temiz-su senaryoları yeterli olduğundan indirilmemiştir.

## Diğer katmanlar (opsiyonel, ileri doğrulama)

- **KATMAN 1 (analitik):** Stoker + Ritter zaten `validation/stoker_analytic.py`
  içinde kodlu. Daha fazlası için SWASHES/pyswashes (swashes C++ ikilisi gerekir).
- **KATMAN 3 (gerçek vaka):** Ürkmez (`cases/urkmez.json`) + Malpasset/Toce
  benchmark'ları — bkz. `docs/kaynaklar.md`.
