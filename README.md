# Parametrik PINN ile Baraj Yıkılma (Dam-Break) Taşkın Analizi

Saint-Venant denklemlerine dayalı **parametrik** bir Physics-Informed Neural
Network (PINN) ile baraj yıkılması **sonrası** mansap taşkın yayılımını modelleyen
BAP araştırma projesi.

Klasik araçlar (HEC-RAS, FLO-2D) doğru ama yavaştır — tek senaryo saatler sürer.
Bir kez eğitilen bu model, **her senaryoyu saniyeler içinde** çözer.

## Projeyi Ayıran 3 Şey

1. **Parametrik.** Ağ tek bir senaryoyu değil, bir senaryo **ailesini** öğrenir:

   ```
   h, u = f(x, t, B, t_f, h0, n)
   ```

   `x,t` uzay-zaman; `B` kırılma genişliği, `t_f` kırılma süresi, `h0` rezervuar
   seviyesi, `n` Manning katsayısı. Bir kez eğit → herhangi bir `(B, t_f, h0, n)`
   için yeniden eğitmeden anında çöz. **Asıl katkı budur.**
2. **Eğitim verisi gerektirmez.** PINN, Saint-Venant denkleminden öğrenir; ölçüm
   veri seti ile eğitilmez. Doğrulama için ise hem **analitik** (Stoker/Ritter) hem
   de gerçek **laboratuvar ölçümleri** (Vosoughi vd. 2021) kullanılır — bkz. aşağıda.
3. **Gerçek zamanlı + yorumlanabilir.** PINN hızı + LLM tabanlı doğal dil afet raporu.

## Fizik

**1D Saint-Venant (sığ su) denklemleri:**

```
süreklilik :  ∂h/∂t + ∂(hu)/∂x = 0
momentum   :  ∂(hu)/∂t + ∂/∂x(hu² + gh²/2) = gh(S0 - Sf)
Manning    :  Sf = n²·u·|u| / h^(4/3)
```

**Froehlich (2008) kırılma parametreleri (overtopping):**

```
B_ave = 0.27·K0·Vw^0.32·hb^0.04        (K0 = 1.3)
t_f   = 0.0176·√(Vw / (g·hb²))          (saat)
```

## Kademeli Geliştirme Stratejisi

| Faz | İçerik | Durum |
|-----|--------|-------|
| **Faz 1** | Tek senaryo 1D PINN → Stoker ile doğrula | ✅ Tam (bu repoda) |
| **Faz 2** | 1–2 parametre ekle (ör. yalnızca `h0`) → parametrik mantığı kanıtla | ⏭️ Sonraki |
| **Faz 3** | Tam parametrik uzay `(B, t_f, h0, n)` | 🧩 İskelet hazır |
| **Faz 4** | Ürkmez + diğer barajlarda test | 🧩 `cases/` hazır |

Şok dalgası + parametrik uzay convergence'ı zor olduğundan **doğrudan Faz 3'e
atlanmaz**.

## Klasör Yapısı

```
PINN-BAP/
├── config/param_ranges.yaml        # Parametre aralıkları + Faz 1 ayarları
├── src/
│   ├── physics/
│   │   ├── saint_venant.py         # 1D Saint-Venant residual (DeepXDE, Manning dahil)
│   │   └── froehlich.py            # Kırılma parametreleri (Ürkmez ile test)
│   ├── pinn/
│   │   ├── pinn_baseline.py        # Faz 1: tek senaryo PINN
│   │   ├── pinn_parametric.py      # Faz 3: parametrik PINN (iskelet)
│   │   └── sampling.py             # Latin Hypercube örnekleme
│   ├── data/parse_vosoughi2021.py  # Deneysel veri seti indir + ayrıştır (Katman 2)
│   └── llm/reporter.py             # PINN → Türkçe afet raporu (iskelet)
├── validation/
│   ├── stoker_analytic.py          # Stoker + Ritter analitik çözüm (ground truth)
│   ├── compare_baseline.py         # Faz 1: PINN vs Stoker
│   ├── compare_experimental.py     # Katman 2: deney vs analitik (RMSE)
│   └── compare_parametric.py       # Faz 3: çoklu parametre testi (iskelet)
├── data/                           # Doğrulama verileri (raw + processed CSV)
├── cases/urkmez.json               # Ürkmez parametreleri (+ yeni baraj şablonu)
├── notebooks/01_baseline_dambreak.ipynb
├── results/                        # Grafikler ve modeller
└── docs/kaynaklar.md               # Kaynak linkleri
```

## Kurulum

`torch`, `numpy`, `scipy`, `matplotlib` genelde kuruludur. Eksik olan tek paket
genellikle DeepXDE'dir:

```bash
pip install deepxde pyyaml          # veya: pip install -r requirements.txt
```

DeepXDE **PyTorch backend** ile kullanılır. Çalıştırırken:

```bash
export DDE_BACKEND=pytorch          # veya komut başına: DDE_BACKEND=pytorch python ...
```

(Scriptler backend'i kod içinde de ayarlar; macOS/Apple Silicon'da CPU kullanılır.)

## Çalıştırma — Faz 1

```bash
# 1) Kırılma parametreleri (Ürkmez doğrulaması)
python src/physics/froehlich.py

# 2) Stoker analitik çözüm (profil grafiği)
python validation/stoker_analytic.py

# 3) Baseline PINN — hızlı uçtan uca test (birkaç dk)
DDE_BACKEND=pytorch python src/pinn/pinn_baseline.py --smoke
#    ...veya tam eğitim (doğrulama için):
DDE_BACKEND=pytorch python src/pinn/pinn_baseline.py

# 4) PINN vs Stoker karşılaştırması (hata + grafik)
python validation/compare_baseline.py
```

Çıktılar `results/` altına yazılır (`baseline_vs_stoker.png`, bağıl L2 hatası vb.).
`--smoke` modunda hata yüksektir (convergence beklenmez); doğrulama için tam eğitim
çalıştırın. Demo için: `notebooks/01_baseline_dambreak.ipynb`.

## Doğrulama Verisi (3 Katman)

PINN eğitim verisi kullanmaz; aşağıdaki veriler **kanıt** içindir:

1. **Analitik (exact):** Stoker (ıslak) + Ritter (kuru) — `validation/stoker_analytic.py`.
2. **Laboratuvar (gerçek ölçüm):** Vosoughi vd. (2021) açık erişimli deneysel veri
   seti (Mendeley, CC BY 4.0). İndir + karşılaştır:

   ```bash
   python src/data/parse_vosoughi2021.py       # veri indir + CSV üret (data/processed/)
   python validation/compare_experimental.py   # deney vs analitik + RMSE (~2 cm)
   ```

   Detay/atıf: [`data/README.md`](data/README.md). Erken zamanlarda deney analitiği
   izler; geç zamandaki fark **taban sürtünmesi**dir — PINN'in Manning terimi tam da
   bunu yakalar (projenin motivasyonu).
3. **Gerçek vaka:** Ürkmez (`cases/urkmez.json`) + Malpasset/Toce (`docs/kaynaklar.md`).

## 2D Genişletme (Faz 2)

Baraj yıkılması sonrası su ovaya **2 boyutlu** yayılır (hangi mahalle batar?).
1D modülleri korunur; 2D paralel modüller olarak eklenir. Fizik: 2D Saint-Venant
`h, u, v = f(x, y, t)` (`src/physics/saint_venant_2d.py`).

| Faz | İçerik | Durum |
|-----|--------|-------|
| **2A** | 2D tek senaryo radyal dam-break → radyal referansla doğrula | ✅ Tam |
| **2B** | Ürkmez 2D + FLO-2D karşılaştırma | 🧩 İskelet |
| **2C** | Parametrik 2D `(x,y,t,B,t_f,h0,n)` = 7 girdi | 🧩 İskelet |

2D'de tam analitik yoktur. Doğrulama iki referansla yapılır:
- **Düzlemsel Stoker eşlemesi** (`validation/radial_reference.py`) — hızlı, yalnızca
  erken-zaman/cephe civarı geçerli (1/r terimi yok).
- **Radyal FV çözücü** (`validation/radial_fv_reference.py`) — 1/r kaynak terimli
  silindirik sığ su, **merkez dahil geçerli, yayınlanabilir referans**.

**Sonuç (GPU'da tam eğitim, FV referansına karşı):** radyal L2 **%2.9–4.8**
(t=0.2–0.6 s arası), şok cephesi ve merkez boşalması dahil. (Düzlemsel referansla
geç zaman %24 görünür — bu PINN hatası değil, düzlemsel referansın 1/r kusurudur.)

```bash
python src/physics/saint_venant_2d.py            # 2D residual sanity
python validation/radial_fv_reference.py         # gerçek 2D FV referansı (+ düzlemsel kıyas)
DDE_BACKEND=pytorch python src/pinn/pinn_baseline_2d.py --smoke   # 2A hızlı test
python validation/compare_baseline_2d.py         # heatmap + radyal kesit + temiz L2
```

Ayar: `config/param_ranges_2d.yaml`. Çıktılar: `results/2d/`. **2D tam eğitim CPU'da
ağırdır** — GPU/Colab önerilir (DeepXDE + PyTorch; Mumtaz 2025 Colab referans).

## Sonraki Adım (Faz 3)

`pinn_parametric.py` ağ girdisine `(B, t_f, h0, n)` ekleyen iskeleti içerir;
`sampling.py` Latin Hypercube ile parametre uzayını örnekler. IC/BC'nin parametreye
bağlı kısımları (rezervuar seviyesi `h0`, kırılma hidrografı `B`/`t_f`) Faz 1
doğrulandıktan sonra doldurulacaktır.

## Kaynaklar

Bkz. [`docs/kaynaklar.md`](docs/kaynaklar.md). Ürkmez benchmark: Haltas, Tayfur &
Elci (2016), FLO-2D, ~%14 derinlik hatası.
