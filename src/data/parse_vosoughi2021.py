"""Vosoughi vd. (2021) deneysel dam-break veri setini indir ve ayrıştır.

Açık erişimli (Mendeley Data, CC BY 4.0) bir laboratuvar dam-break veri seti.
12 senaryonun ilk 4'ü TEMİZ SU (sedimentsiz) rezervuarıdır ve doğrudan 1D
Saint-Venant PINN doğrulaması için kullanılır:

    Tablo 1: temiz su, KURU mansap            -> Ritter (kuru-yatak) analitik
    Tablo 2: temiz su, ISLAK mansap  2 cm     -> Stoker (ıslak-yatak) analitik
    Tablo 3: temiz su, ISLAK mansap  4 cm     -> Stoker
    Tablo 4: temiz su, ISLAK mansap  5 cm     -> Stoker

Her senaryo: 20 ölçüm konumu (L = 0..552 cm) x 15 zaman anı (0.04..6.0 s),
rezervuar yüksekliği H = 30 cm. Serbest su yüzeyi seviyesi cm cinsinden.

Kaynak / atıf:
    Vosoughi, F., Rakhshandehroo, G., Nikoo, M.R. (2021). "Experimental dataset
    on water levels in studying the influences of dry- and wet-bed downstream
    conditions ..." Data in Brief. Mendeley Data, v3.
    DOI: 10.17632/nc573y67tp.3

Çalıştırma:
    python src/data/parse_vosoughi2021.py        # indir (gerekirse) + CSV üret
"""
from __future__ import annotations

import csv
import os
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(ROOT, "data", "raw", "vosoughi2021_dataset.docx")
OUT_DIR = os.path.join(ROOT, "data", "processed")

# Mendeley Data v3 doğrudan indirme bağlantısı (tek dosya: Dataset.docx)
DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/nc573y67tp/files/"
    "10b97bf4-27f1-4ad3-ae32-0296c36b4f80/file_downloaded"
)

RESERVOIR_HEIGHT_CM = 30.0

# Temiz-su senaryolarının metadata'sı (docx'teki Tablo 1-4 = indeks 0-3)
CLEARWATER = [
    {"table": 1, "downstream": "dry",  "h_down_cm": 0.0, "slug": "clearwater_dry"},
    {"table": 2, "downstream": "wet",  "h_down_cm": 2.0, "slug": "clearwater_wet2cm"},
    {"table": 3, "downstream": "wet",  "h_down_cm": 4.0, "slug": "clearwater_wet4cm"},
    {"table": 4, "downstream": "wet",  "h_down_cm": 5.0, "slug": "clearwater_wet5cm"},
]


def download_if_missing() -> None:
    if os.path.exists(RAW_PATH):
        return
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    print(f"[parse] veri seti indiriliyor: {DOWNLOAD_URL}")
    urllib.request.urlretrieve(DOWNLOAD_URL, RAW_PATH)
    print(f"[parse] kaydedildi: {RAW_PATH}")


def _cell_text(tc) -> str:
    return "".join(t.text or "" for t in tc.iter(f"{W}t")).strip()


def read_tables(path: str = RAW_PATH) -> list[list[list[str]]]:
    """docx içindeki tüm tabloları satır/hücre matrisleri olarak döndürür."""
    xml = zipfile.ZipFile(path).read("word/document.xml")
    body = ET.fromstring(xml).find(f"{W}body")
    tables = []
    for el in body:
        if el.tag == f"{W}tbl":
            tables.append([
                [_cell_text(tc) for tc in tr.findall(f"{W}tc")]
                for tr in el.findall(f"{W}tr")
            ])
    return tables


def _to_float(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def parse_clearwater(tables) -> list[dict]:
    """Tablo 1-4'ü uzun-format (long) kayıtlara çevirir.

    Her kayıt: {scenario, downstream, h_down_cm, L_cm, time_s, level_cm}.
    Tablo yapısı: satır1 = zamanlar (sütun 1..15), satır3+ = konumlar (sütun0=L).
    """
    records = []
    for meta in CLEARWATER:
        tbl = tables[meta["table"] - 1]
        times = [_to_float(c) for c in tbl[1][1:]]            # 15 zaman (s)
        for row in tbl[3:]:                                    # 20 konum satırı
            L = _to_float(row[0])
            if L is None:
                continue
            for t, val in zip(times, row[1:]):
                level = _to_float(val)
                if t is None or level is None:
                    continue
                records.append({
                    "scenario": meta["slug"],
                    "downstream": meta["downstream"],
                    "h_down_cm": meta["h_down_cm"],
                    "L_cm": L,
                    "time_s": t,
                    "level_cm": level,
                })
    return records


def write_csvs(records: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Birleşik uzun-format CSV
    combined = os.path.join(OUT_DIR, "vosoughi2021_clearwater.csv")
    fields = ["scenario", "downstream", "h_down_cm", "L_cm", "time_s", "level_cm"]
    with open(combined, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # 2) Senaryo başına ayrı CSV
    per = {}
    for r in records:
        per.setdefault(r["scenario"], []).append(r)
    for slug, rows in per.items():
        path = os.path.join(OUT_DIR, f"vosoughi2021_{slug}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    print(f"[parse] {len(records)} kayıt yazıldı:")
    print(f"        birleşik : {combined}")
    for slug in per:
        print(f"        senaryo  : vosoughi2021_{slug}.csv  ({len(per[slug])} kayıt)")


def main() -> None:
    download_if_missing()
    tables = read_tables()
    print(f"[parse] docx içinde {len(tables)} tablo bulundu (12 senaryo beklenir).")
    records = parse_clearwater(tables)
    write_csvs(records)
    print(f"[parse] ✓ Temiz-su senaryoları hazır. Rezervuar H = {RESERVOIR_HEIGHT_CM} cm.")


if __name__ == "__main__":
    main()
