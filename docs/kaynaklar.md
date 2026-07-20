# Kaynaklar

Projede adapte edilen / referans alınan kod, veri ve makale kaynakları.

---

## 🎯 Veri Stratejisi — 3 Katman (kanıt yapısı)

PINN eğitim verisi gerektirmez; aşağıdaki veriler **doğrulama/kanıt** içindir.

| Katman | Amaç | Durum |
|---|---|---|
| **1. Analitik** | PINN'in DOĞRU çözdüğünü ispatlar (exact) | ✅ Stoker + Ritter kodlu (`validation/stoker_analytic.py`) |
| **2. Laboratuvar** | GERÇEK ölçümle karşılaştırma | ✅ **Vosoughi vd. 2021 indirildi + ayrıştırıldı** (`data/`) |
| **3. Gerçek vaka** | Saha uygulaması (case study) | 🧩 Ürkmez (`cases/`) + Malpasset/Toce (aşağıda) |

### ✅ İndirilen veri seti (Katman 2)
**Vosoughi, Rakhshandehroo, Nikoo (2021)** — açık erişimli laboratuvar dam-break.
Temiz-su (sedimentsiz) senaryoları: kuru + ıslak (2/4/5 cm) mansap, H=30 cm,
20 konum × 15 zaman. Deney vs analitik early-time RMSE ≈ 2 cm.
- Mendeley Data v3, **DOI 10.17632/nc573y67tp.3**, CC BY 4.0
- https://data.mendeley.com/datasets/nc573y67tp/3
- Ayrıntı + atıf: bkz. [`data/README.md`](../data/README.md)

### Analitik araçlar (Katman 1, opsiyonel ek)
- **SWASHES / pyswashes** (CSV/NumPy üreten analitik çözüm kütüphanesi):
  https://github.com/lrntct/pyswashes — `pip install pyswashes` ama `swashes` C++
  ikilisi gerekir (conda-forge). Stoker/Ritter zaten kodlu olduğundan opsiyonel.
- SWASHES makalesi: https://arxiv.org/abs/1110.0288 · HAL: https://hal.science/hal-00694195v3

---

## PINN Kodları (adapte edilecek)
- **Mumtaz 2025** (PLOS ONE, DeepXDE + Colab, 1D/2D dam-break):
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332694
- **Tian 2025** (2D SWE PINN, topografya): https://arxiv.org/abs/2501.11372
- **Qi 2024** (2D SWE PINN, etiketsiz veri):
  https://www.sciencedirect.com/science/article/pii/S0022169424006589
- **DeepXDE framework**: https://github.com/lululxvi/deepxde
- **DeepXDE SWE/PINN makale listesi**:
  https://github.com/lululxvi/deepxde/blob/master/docs/user/research.rst

## Analitik Doğrulama (Faz 1)
- **SWASHES** makale: https://arxiv.org/abs/1110.0288
- **pyswashes** (Python): https://github.com/lrntct/pyswashes
- **Stoker (1948/1957)** DOI: https://doi.org/10.1002/cpa.3160010101

## Laboratuvar Verisi (ileri doğrulama, opsiyonel)
- **CADAM** raporu (ücretsiz PDF): http://www.ib-nujic.de/Dammbruch/CADAM_final.pdf
- **IMPACT** veritabanı (JHR):
  https://www.tandfonline.com/doi/abs/10.1080/00221686.2007.9521827
- **Toce River** (HEC-RAS, PMC): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8256810/
- **Malpasset** 2D benchmark (arXiv): https://arxiv.org/pdf/2410.05968

## Ürkmez Test Vakası
- **Haltas, Tayfur & Elci (2016)** — parametreler Tablo 1'de (FLO-2D, ~%14 derinlik hatası)
- **WebPlotDigitizer** (grafik → sayı): https://automeris.io/WebPlotDigitizer/

## Karşılaştırma Modelleri (opsiyonel, veri-odaklı)
- **FloodSformer** (transformer): https://zenodo.org/records/16262014
- **ESN/LSTM dam-break**: https://github.com/lcl1527/dam-break-ESN-LSTM

## Fizik / Kırılma Denklemleri
- **Froehlich, D.C. (2008)** — "Embankment Dam Breach Parameters and Their
  Uncertainties." J. Hydraulic Engineering, 134(12), 1708-1721.

## 2D Genişletme (Faz 2)
- **Tian vd. (2025)** — 2D SWE PINN, radyal dam-break dahil doğrulama:
  https://arxiv.org/abs/2501.11372
- **Mumtaz vd. (2025)** — DeepXDE + Colab (GPU) ile 2D dam-break, tam eğitim için
  referans: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332694
- **2D doğrulama:** (i) düzlemsel Stoker eşlemesi (`validation/radial_reference.py`,
  hızlı sanity, erken-zaman) ve (ii) **radyal FV çözücü** (`validation/radial_fv_reference.py`,
  1/r kaynak terimli silindirik sığ su — merkez dahil geçerli, yayınlanabilir referans).
  PINN tam eğitimde FV'ye karşı radyal L2 %2.9–4.8. Ayrıca Ürkmez FLO-2D
  karşılaştırması (`validation/compare_urkmez_flo2d.py`, iskelet).
- **2D lab verisi (ileri doğrulama):** Toce River / Soares-Frazão — CADAM raporu +
  MDPI Water review (yukarıdaki "Laboratuvar Verisi" bölümü).
