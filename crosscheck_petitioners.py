"""Add an independent second-coder label (model2_petitioner_sex) to the hand-label sample.

Reads data/output/handlabel_petitioners.csv (unchanged) and writes
data/output/crosscheck_petitioners.csv. Stops if the ids do not match exactly.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "output" / "handlabel_petitioners.csv"
DST = ROOT / "data" / "output" / "crosscheck_petitioners.csv"

F, M, U = "female", "male", "unknown"

LABELS = {
    "co_800": F, "co_4629": F, "co_918": F, "co_5815": M, "co_2245": U,
    "co_1956": M, "co_2914": F, "co_4004": F, "co_1641": F, "co_5883": M,
    "co_3398": M, "co_2932": F, "co_4909": F, "co_4013": F, "co_5872": M,
    "co_2841": M, "co_4134": M, "co_4225": F, "co_2944": M, "co_919": F,
    "co_1874": F, "co_4471": F, "co_2442": F, "co_5090": F, "co_6668": F,
    "co_6458": U, "co_1901": F, "co_1549": F, "co_4690": U, "co_1608": F,
    "co_302": F, "co_3631": M, "co_5769": U, "co_5922": M, "co_1437": M,
    "co_5847": F, "co_5357": M, "co_4995": U, "co_4110": F, "co_2087": M,
    "co_742": U, "co_832": U, "co_1837": M, "co_292": F, "co_4046": F,
    "co_4333": F, "co_3127": U, "co_1244": F, "co_1416": M, "co_3846": M,
    "co_3064": F, "co_4797": U, "co_2074": M, "co_375": F, "co_1706": F,
    "co_2894": U, "co_2423": M, "co_2937": M, "co_3997": M, "co_1505": M,
    "co_5499": U, "co_4002": F, "co_782": F, "co_4474": F, "co_6651": F,
    "co_3058": F, "co_4884": M, "co_6524": M, "co_3266": F, "co_2056": F,
    "co_1819": F, "co_4798": F, "co_5724": M, "co_4683": M, "co_4560": U,
    "co_2148": U, "co_1953": F, "co_799": M, "co_2328": F, "co_5249": M,
    "co_5025": M, "co_3569": F, "co_2410": F, "co_4404": M, "co_5522": M,
    "co_3748": F, "co_6070": F, "co_3834": U, "co_6147": M, "co_5718": F,
    "co_2775": F, "co_103": F, "co_283": U, "co_5983": U, "co_5120": U,
    "co_3216": M, "co_771": M, "co_2595": M, "co_5508": U, "co_5600": M,
    "co_2154": F, "co_882": F, "co_6325": F, "co_4377": M, "co_4617": F,
    "co_6303": F, "co_4102": F, "co_5982": U, "co_4152": F, "co_1319": M,
    "co_6805": F, "co_2748": M, "co_2075": F, "co_6683": F, "co_3297": F,
    "co_4085": U, "co_1780": F, "co_2070": M, "co_6344": F, "co_1750": F,
    "co_1916": F, "co_1647": F, "co_6764": U, "co_1639": M, "co_5620": M,
    "co_5834": F, "co_275": M, "co_229": M, "co_1930": M, "co_4976": F,
    "co_3087": F, "co_4308": F, "co_2130": U, "co_5573": M, "co_4739": M,
    "co_4802": F, "co_3950": M, "co_543": F, "co_4387": M, "co_6768": U,
    "co_99": U, "co_5543": U, "co_92": M, "co_5652": M, "co_1723": F,
    "co_5671": F, "co_162": F, "co_4726": U, "co_5114": M, "co_5293": F,
    "co_4982": U, "co_1775": F, "co_5037": F, "co_676": F, "co_2733": F,
    "co_1712": F, "co_6748": U, "co_3751": F, "co_1084": M, "co_5307": M,
    "co_496": M, "co_2015": F, "co_6740": U, "co_5258": U, "co_2281": M,
    "co_2221": F, "co_612": F, "co_947": M, "co_3635": F, "co_3707": U,
    "co_4059": F, "co_1503": F, "co_2727": F, "co_587": F, "co_4239": M,
    "co_6673": U, "co_2100": F, "co_3697": F, "co_2456": F, "co_6310": F,
    "co_5360": M, "co_6789": U, "co_951": M, "co_3510": M, "co_4443": M,
    "co_170": M, "co_3739": F, "co_1992": M, "co_3701": F, "co_3571": M,
    "co_4625": F, "co_4254": M, "co_2958": U, "co_6670": M, "co_5316": F,
    "co_149": M, "co_1968": M, "co_2079": F, "co_1769": M, "co_281": F,
}


def main():
    with SRC.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    ids = [r["id"] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    missing = sorted(set(ids) - LABELS.keys())
    extra = sorted(LABELS.keys() - set(ids))
    if dupes or missing or extra:
        sys.exit(f"id mismatch: duplicates={dupes} missing_labels={missing} unknown_ids={extra}")

    out_field = "model2_petitioner_sex"
    if out_field in fields:
        sys.exit(f"{out_field} already present in {SRC}")
    for r in rows:
        r[out_field] = LABELS[r["id"]]

    with DST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + [out_field])
        writer.writeheader()
        writer.writerows(rows)

    counts = {k: sum(r[out_field] == k for r in rows) for k in (F, M, U)}
    print(f"wrote {len(rows)} rows to {DST.relative_to(ROOT)}: {counts}")


if __name__ == "__main__":
    main()
