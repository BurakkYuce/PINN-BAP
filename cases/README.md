# Baraj Vakaları (cases/)

Bu klasör, parametrik PINN'in test edileceği gerçek baraj senaryolarını tutar.
Her vaka, parametre uzayında **bir nokta**dır — model tek bir baraja özel değildir,
bir kez eğitilince her vaka için anında çözüm verir.

## ⚠ ÖNEMLİ: Baraj tipi yıkılma mekanizmasını belirler

Tüm barajlar aynı modellenemez! `dam_type` alanı zorunludur:
- **`earth-fill` / `rockfill` (toprak/kaya dolgu):** KADEMELİ kırılma → Froehlich +
  parametrik PINN (B, t_f, h0, n) uygulanır. `in_scope: true`. Örn: Ürkmez, Korkuteli.
- **`concrete-arch` / `concrete-gravity` (beton):** ANİ yıkılma (temel/yapısal) →
  Froehlich UYGULANMAZ; ani dam-break (t_f≈0) gibi ele alınır, tetik geotekniktir.
  `in_scope: false`. Örn: Oymapınar, Yusufeli.

Ayrıntı: [`../docs/baraj_tipleri.md`](../docs/baraj_tipleri.md).

## Yeni baraj eklemek için şablon

`<baraj_adi>.json` adıyla yeni bir dosya oluştur:

```json
{
  "name": "Örnek Barajı",
  "country": "Türkiye",
  "reference": "Kaynak (yazar, yıl, yöntem)",
  "dam_type": "earth-fill",
  "failure_mechanism": "gradual breach (overtopping)",
  "in_scope": true,
  "failure_mode": "overtopping",
  "reservoir_volume_m3": 0.0,
  "crest_elevation_m": 0.0,
  "final_bed_elevation_m": 0.0,
  "breach_height_m": 0.0,
  "breach_width_avg_m": 0.0,
  "breach_formation_time_hr": 0.0,
  "peak_discharge_m3s": 0.0,
  "manning_n": 0.04,
  "overtopping": true,
  "notes": "Açıklama"
}
```

## Alan açıklamaları

| Alan | Açıklama | Birim |
|------|----------|-------|
| `reservoir_volume_m3` | Kırılma anında rezervuar su hacmi (Vw) | m³ |
| `breach_height_m` | Kırılma yüksekliği (hb) | m |
| `breach_width_avg_m` | Ortalama kırılma genişliği (B) — ölçülmemişse Froehlich ile hesaplanır | m |
| `breach_formation_time_hr` | Kırılma oluşum süresi (t_f) | saat |
| `peak_discharge_m3s` | Pik debi | m³/s |
| `manning_n` | Manning pürüzlülük katsayısı | — |

## Froehlich ile parametre üretimi

`breach_width_avg_m` ve `breach_formation_time_hr` bilinmiyorsa, `Vw` ve `hb`
değerlerinden hesaplanabilir:

```bash
python src/physics/froehlich.py   # Ürkmez örneğini doğrular; fonksiyon: breach_params(Vw, hb)
```

## Mevcut vakalar

| Dosya | Tip | Kapsamda? |
|-------|-----|-----------|
| `urkmez.json` | toprak dolgu | ✅ (referans benchmark, Haltas vd. 2016) |
| `korkuteli.json` | toprak gövde dolgu | ✅ (ikinci dolgu vakası) |
| `oymapinar.json` | beton kemer | ⚠ ayrı rejim (ani dam-break; Froehlich uygulanmaz) |
