"""Tokenizer statistics per model on the prepared FPB data (EN and HU, same sentences).

Usage: python tokstats.py [--config config.yaml]

Words are \\w+ runs (punctuation excluded). Tokens are mapped to words via character offsets of the
full-sentence tokenization (no special tokens), so words are tokenized in context.
  fertility    mean number of tokens per word
  split_rate   share of words split into 2+ tokens
  tok_per_sent mean tokens per sentence (no special tokens)
  max_len      max tokens per sentence incl. special tokens; n_trunc = sentences above config max_len
HU/EN ratios are computed on the same parallel sentences.
Output: results/tokstats.csv
"""
import argparse
import csv
import json
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml
from transformers import AutoTokenizer

WORD = re.compile(r"\w+")


def stats(tok, texts, max_len):
    n_words = n_word_tokens = n_split = n_tokens = n_trunc = longest = 0
    for text in texts:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = [(s, e) for s, e in enc["offset_mapping"] if e > s]
        n_tokens += len(enc["input_ids"])
        full = len(tok(text)["input_ids"])  # incl. special tokens
        longest = max(longest, full)
        n_trunc += full > max_len
        for m in WORD.finditer(text):
            k = sum(1 for s, e in offs if s < m.end() and e > m.start())
            n_words += 1
            n_word_tokens += k
            n_split += k > 1
    return {
        "fertility": n_word_tokens / n_words,
        "split_rate": n_split / n_words,
        "tok_per_sent": n_tokens / len(texts),
        "max_len": longest,
        "n_trunc": n_trunc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(cfg["data"]["out_dir"], "fpb_split.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    out = []
    for m in cfg["models"]:
        tok = AutoTokenizer.from_pretrained(m["path"])
        assert tok.is_fast, f"{m['name']}: offsets need a fast tokenizer"
        r = {"model": m["name"]}
        for lang in ("en", "hu"):
            for k, v in stats(tok, [x[lang] for x in rows], cfg["max_len"]).items():
                r[f"{lang}_{k}"] = v
        r["hu_en_fertility"] = r["hu_fertility"] / r["en_fertility"]
        r["hu_en_tokens"] = r["hu_tok_per_sent"] / r["en_tok_per_sent"]
        out.append(r)

    os.makedirs("results", exist_ok=True)
    path = "results/tokstats.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    # Short summary that can be typed back by hand
    print(f"TOK n_sent={len(rows)} max_len_cfg={cfg['max_len']}")
    print("TOK model fert_en fert_hu split_en split_hu tok_hu/en maxlen_en maxlen_hu trunc_en trunc_hu")
    for r in out:
        print(f"TOK {r['model']} {r['en_fertility']:.2f} {r['hu_fertility']:.2f} {r['en_split_rate']:.3f} "
              f"{r['hu_split_rate']:.3f} {r['hu_en_tokens']:.2f} {r['en_max_len']} {r['hu_max_len']} "
              f"{r['en_n_trunc']} {r['hu_n_trunc']}")
    print(f"TOK saved {path}")


if __name__ == "__main__":
    main()
