# Baraj Tipi ve Yıkılma Mekanizması — Modelleme Kapsamı

> **Kritik not:** Bulunan tüm barajlar aynı şekilde modellenemez. Yıkılma mekanizması
> baraj **tipine** göre kökten değişir. Bu proje (Froehlich kademeli kırılma +
> parametrik PINN) **toprak/kaya dolgu (embankment) barajlar** içindir.

## İki rejim

### 1) Toprak/Kaya Dolgu (Embankment) — KADEMELİ kırılma  ✅ bu projenin kapsamı
- Yıkılma, gövdenin **aşamalı erozyonu** ile olur (overtopping veya piping).
- Kırılma zamanla büyür: genişlik **B**, oluşum süresi **t_f**.
- **Froehlich (2008)** denklemleri tam bunu modeller → bizim `src/physics/froehlich.py`.
- Parametrik PINN girdileri **(B, t_f, h0, n)** doğrudan bu rejime aittir.
- IC/giriş koşulu: zamana bağlı kırılma hidrografı (pik debi Froehlich'ten).

### 2) Beton (Gravite / Kemer) — ANİ yıkılma  ⚠ farklı rejim, ayrı ele alınır
- Beton gövde çok güçlüdür; sorun gövdede değil **temel/zemin mekaniği**ndedir
  (temel kayması, yan duvar/abutment göçmesi, devrilme).
- Yıkılma **ani / tek parça**dır — kademeli kırılma YOKTUR (t_f ≈ 0).
- Hidrolik olarak **ani (instantaneous) dam-break** gibi modellenir: rezervuar bir
  anda boşalır → klasik **Ritter/Stoker** ani dam-break (bizde `validation/stoker_analytic.py`
  zaten bunu temsil eder; t_f→0).
- Yıkılma **tetikleyicisi** geoteknik/yapısal bir problemdir → bu hidrolik PINN'in
  kapsamı DIŞINDA (ayrı zemin mekaniği analizi gerektirir).
- **Froehlich denklemleri beton barajlara UYGULANMAZ.**

## Sınıflandırma (doğrulanmış)

| Baraj | Tip | Yükseklik | Mekanizma | Bu modelde? |
|-------|-----|-----------|-----------|-------------|
| **Ürkmez** | toprak dolgu | ~45.9 m | kademeli kırılma (overtopping) | ✅ Evet (Froehlich + PINN) |
| **Korkuteli** | toprak gövde dolgu | 70 m | kademeli kırılma | ✅ Evet (earth-fill) |
| **Oymapınar** | beton çift-kavisli kemer | 185 m | ani / temel-yapısal | ⚠ Hayır (ani dam-break + geoteknik) |
| **Yusufeli** | beton çift-kavisli kemer | 275 m | ani / temel-yapısal | ⚠ Hayır (ani dam-break + geoteknik) |

Kaynaklar: Korkuteli — [Vikipedi](https://tr.wikipedia.org/wiki/Korkuteli_Baraj%C4%B1)
(toprak gövde dolgu, 70 m, göl 47.5 hm³); Oymapınar / Yusufeli —
[Vikipedi](https://tr.wikipedia.org/wiki/Oymap%C4%B1nar_Baraj%C4%B1_ve_Hidroelektrik_Santrali),
[Yusufeli](https://www.limak.com.tr/sektorler/insaat/projeler/tamamlanan-tum-projeler/barajlar/yusufeli-baraji-ve-hes)
(beton çift-kavisli kemer).

## Modelleme sonucu (özet)

| | Embankment (toprak/kaya) | Beton (gravite/kemer) |
|---|---|---|
| Kırılma | kademeli (B, t_f) | ani (t_f ≈ 0) |
| Parametre modeli | Froehlich → PINN (B,t_f,h0,n) | uygulanmaz |
| Hidrolik IC | kırılma hidrografı | tam ani boşalma (Ritter/Stoker) |
| Tetikleyici | hidrolik/erozyon | geoteknik/yapısal (kapsam dışı) |
| Örnek | Ürkmez, Korkuteli | Oymapınar, Yusufeli |

## Yapılacaklar / hocaya sorulacak
- Çalışma **toprak dolgu** barajlarla sınırlandırılırsa (Ürkmez + Korkuteli) model
  tutarlıdır ve doğrudan uygulanır.
- Beton barajlar dahil edilecekse: ya yalnızca **ani dam-break** (t_f→0) senaryosu
  olarak ele alınır (hidrolik), ya da yıkılma tetiği için ayrı **geoteknik** modül
  gerekir — bu, hidrolik PINN'in dışındadır. **Hocaya bu kapsam kararı sorulmalı.**
- `cases/*.json` dosyalarında her baraj için `dam_type` ve `failure_mechanism`
  alanları eklendi; kod bu alana göre uygun rejimi seçmelidir.
