"""Prepare FPB data: manual fixes, deduplication, stratified train/test split, learning-curve subsets.

Usage: python prepare_data.py [--config config.yaml]

Outputs (in data.out_dir):
  fpb_split.jsonl  one row per kept sentence: id, label, label_name, en, hu, split ("train"/"test"),
                   allagree (exact EN text match with the original FPB Sentences_AllAgree.txt)
  subsets.json     {seed: {size: [train ids]}}, nested stratified subsets (100 in 250 in 500 in 1000)
  hu_native.txt    native Hungarian economic news sentences (one per line), for BPC only
The ids are the row indices of fpb.jsonl, identical for EN and HU.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter

import yaml
from sklearn.model_selection import train_test_split


def norm(s):
    # Normalization used only for near-duplicate detection
    return re.sub(r"\W+", "", s.lower().replace("n't", "not"))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_xlsx_column(path, sheet, column):
    # Minimal stdlib xlsx reader (avoids an openpyxl dependency): returns the values of one column, by header name.
    import zipfile
    import xml.etree.ElementTree as ET
    M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(M + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(M + "si")]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    sheets = {s.get("name"): rels[s.get(R + "id")] for s in ET.fromstring(z.read("xl/workbook.xml")).iter(M + "sheet")}
    target = sheets[sheet].lstrip("/").removeprefix("xl/")
    table = []
    for row in ET.fromstring(z.read("xl/" + target)).iter(M + "row"):
        cells = {}
        for c in row.findall(M + "c"):
            v = c.find(M + "v")
            if v is not None:
                cells[re.match(r"[A-Z]+", c.get("r")).group()] = shared[int(v.text)] if c.get("t") == "s" else v.text
        table.append(cells)
    col = next(k for k, v in table[0].items() if v == column)
    return [r[col] for r in table[1:] if r.get(col)]


def label_counts(rows):
    c = Counter(r["label"] for r in rows)
    return "/".join(str(c[k]) for k in (0, 1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]

    with open(d["input"], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    n_in = len(rows)
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == n_in, "duplicate ids in input"

    # 1) Manual resolution of exact English duplicates
    with open(d["fixes"], encoding="utf-8") as f:
        fixes = json.load(f)["groups"]
    drop = set()
    for g in fixes:
        keep = by_id[g["keep"]]
        assert keep["label_name"] == g["label"], g
        for i in g["drop"]:
            assert by_id[i]["en"] == keep["en"], g
            drop.add(i)
        if "hu" in g:
            assert g["hu"] != "TODO", f"missing Hungarian text for id {g['keep']} in fixes file"
            keep["hu"] = g["hu"]
    rows = [r for r in rows if r["id"] not in drop]

    # 2) Automatic near-duplicate removal (normalized EN or HU), keeping the lowest id.
    # A label conflict stops the script: resolve it manually in the fixes file.
    seen, near, kept = {}, [], []
    for r in sorted(rows, key=lambda r: r["id"]):
        keys = [("en", norm(r["en"])), ("hu", norm(r["hu"]))]
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit is not None:
            assert hit["label"] == r["label"], f"label conflict: {hit['id']} vs {r['id']}"
            near.append((hit["id"], r["id"]))
            continue
        for k in keys:
            seen[k] = r
        kept.append(r)
    rows = kept

    # 3) Fixed-seed stratified train/test split
    ids = [r["id"] for r in rows]
    train_ids, test_ids = train_test_split(
        ids, test_size=d["test_size"], stratify=[r["label"] for r in rows], random_state=d["split_seed"]
    )
    test_set = set(test_ids)
    with open(d["allagree"], encoding="latin-1") as f:
        allagree = {line.rstrip("\r\n").rsplit("@", 1)[0].strip() for line in f if line.strip()}
    for r in rows:
        r["split"] = "test" if r["id"] in test_set else "train"
        r["allagree"] = r["en"] in allagree

    # 4) Nested stratified learning-curve subsets per seed
    lab = {r["id"]: r["label"] for r in rows}
    subsets = {}
    for seed in cfg["seeds"]:
        cur = sorted(train_ids)
        subsets[str(seed)] = {}
        for n in sorted(d["curve_sizes"], reverse=True):
            cur, _ = train_test_split(cur, train_size=n, stratify=[lab[i] for i in cur], random_state=seed)
            cur = sorted(cur)
            subsets[str(seed)][str(n)] = cur

    os.makedirs(d["out_dir"], exist_ok=True)
    split_path = os.path.join(d["out_dir"], "fpb_split.jsonl")
    sub_path = os.path.join(d["out_dir"], "subsets.json")
    with open(split_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(sub_path, "w", encoding="utf-8") as f:
        json.dump(subsets, f, sort_keys=True)

    # 5) Native Hungarian sentences for BPC: whitespace-normalized, deduplicated, one per line
    native = [" ".join(s.replace("\xa0", " ").split()) for s in read_xlsx_column(d["native"], d["native_sheet"], "text")]
    native = [s for s in dict.fromkeys(native) if s]
    native_path = os.path.join(d["out_dir"], "hu_native.txt")
    with open(native_path, "w", encoding="utf-8") as f:
        f.write("\n".join(native) + "\n")

    # Short summary that can be typed back by hand (label order: neg/neu/pos)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"PREP in={n_in} fix_drop={len(drop)} near_drop={len(near)} rows={len(rows)}")
    print(f"PREP near_pairs(kept,dropped)={near}")
    print(f"PREP train={len(train)} ({label_counts(train)}) test={len(test)} ({label_counts(test)})")
    aa_test = [r for r in test if r["allagree"]]
    print(f"PREP allagree file={len(allagree)} kept={sum(r['allagree'] for r in rows)} "
          f"test={len(aa_test)} ({label_counts(aa_test)})")
    for seed, s in subsets.items():
        print(f"PREP seed={seed} " + " ".join(f"{n}:{label_counts([by_id[i] for i in v])}" for n, v in s.items()))
    print(f"SHA fpb_split.jsonl {sha256(split_path)[:16]}")
    print(f"PREP native sentences={len(native)} chars={sum(map(len, native))}")
    print(f"SHA subsets.json {sha256(sub_path)[:16]}")
    print(f"SHA hu_native.txt {sha256(native_path)[:16]}")


if __name__ == "__main__":
    main()
