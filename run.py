#!/usr/bin/env python
"""Experiment runner: fine-tuning (C1/C2 incl. learning curve), zero-shot C0 and BPC.

Each task writes <out>/<task_id>/preds.jsonl and metrics.json. A task is done when metrics.json exists,
so the script can be restarted at any time and skips finished tasks. Several workers share the task list:
a worker claims a task by atomically creating <task_id>/RUNNING. A failed task leaves ERROR.txt.

Tasks (from config.yaml):
  ft   model x train_lang (en/hu) x size (curve sizes + full) x seed; tested on both EN and HU test sets
  c0   decoders only: (test en, prompt en), (test hu, prompt hu), (test hu, prompt en); label log-probs
  bpc  decoders only: bits per character on the EN and HU test sentences and on native HU news sentences

Server, one worker per GPU:
  nohup python run.py --device cuda:0 > worker0.log 2>&1 &
  nohup python run.py --device cuda:1 > worker1.log 2>&1 &
  python run.py --status
After a crash or kill, with no worker running: python run.py --clear_stale (also retries failed tasks)

Smoke test (Mac, CPU):
  python run.py --device cpu --models qwen3-0.6b,hubert --sizes 100 --seeds 1 \
      --max_steps 3 --limit_test 30 --out results_smoke
"""
import os

# Must be set before transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import gc
import json
import math
import random
import re
import time
import traceback

import numpy as np
import torch
import transformers
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

# Label order matches the data: 0=negative, 1=neutral, 2=positive
LABELS = {"en": ["negative", "neutral", "positive"], "hu": ["negatív", "semleges", "pozitív"]}
PROMPTS = {
    "en": ("Classify the sentiment of the following financial news sentence as negative, neutral or positive.\n\n"
           "Sentence: {text}\n\nAnswer with one word."),
    "hu": ("Osztályozd a következő gazdasági hírmondat hangulatát: negatív, semleges vagy pozitív.\n\n"
           "Mondat: {text}\n\nEgyetlen szóval válaszolj."),
}
CONTENT_FREE = "N/A"  # input for the contextual calibration prior in C0


# ---------------------------------------------------------------- tasks

def task_id(t):
    return "__".join(str(v) for v in t.values())


def build_tasks(cfg, args):
    sizes = [str(s) for s in cfg["data"]["curve_sizes"]] + ["full"]
    tasks = []
    for m in cfg["models"]:
        if m["kind"] == "decoder":
            tasks.append({"type": "bpc", "model": m["name"]})
            for test_lang, prompt_lang in (("en", "en"), ("hu", "hu"), ("hu", "en")):
                tasks.append({"type": "c0", "model": m["name"], "test_lang": test_lang, "prompt_lang": prompt_lang})
        for lang in ("en", "hu"):
            for size in sizes:
                for seed in cfg["seeds"]:
                    tasks.append({"type": "ft", "model": m["name"], "train_lang": lang, "size": size, "seed": seed})
    if args.only:
        tasks = [t for t in tasks if t["type"] in args.only.split(",")]
    if args.models:
        tasks = [t for t in tasks if t["model"] in args.models.split(",")]
    if args.sizes:
        tasks = [t for t in tasks if t["type"] != "ft" or t["size"] in args.sizes.split(",")]
    if args.seeds:
        tasks = [t for t in tasks if t["type"] != "ft" or str(t["seed"]) in args.seeds.split(",")]
    for t in tasks:
        t["id"] = task_id(t)
    return tasks


def claim(run_dir):
    os.makedirs(run_dir, exist_ok=True)
    if any(os.path.exists(os.path.join(run_dir, f)) for f in ("metrics.json", "ERROR.txt")):
        return False
    try:
        fd = os.open(os.path.join(run_dir, "RUNNING"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"pid={os.getpid()} start={time.ctime()}".encode())
    os.close(fd)
    return True


def task_state(run_dir):
    for f, s in (("metrics.json", "done"), ("ERROR.txt", "failed"), ("RUNNING", "running")):
        if os.path.exists(os.path.join(run_dir, f)):
            return s
    return "pending"


# ---------------------------------------------------------------- helpers

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))


def ece_score(conf, correct, n_bins=15):
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def metrics(gold, probs):
    gold = np.asarray(gold)
    pred = probs.argmax(1)
    return {
        "macro_f1": float(f1_score(gold, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "acc": float(accuracy_score(gold, pred)),
        "f1_per_class": [float(x) for x in f1_score(gold, pred, average=None, labels=[0, 1, 2], zero_division=0)],
        "confusion": confusion_matrix(gold, pred, labels=[0, 1, 2]).tolist(),
        "ece": ece_score(probs.max(1), pred == gold),
        "n": int(len(gold)),
    }


def evaluate(test, probs):
    gold = np.array([r["label"] for r in test])
    aa = np.array([r["allagree"] for r in test])
    return {"all": metrics(gold, probs), "allagree": metrics(gold[aa], probs[aa])}


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


@torch.no_grad()
def cont_logprob(model, seqs, n_prompt, device, pad_id):
    """Sum of log-probs of seq[n_prompt:] given seq[:n_prompt], for each sequence (right padded batch)."""
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    ids, att = ids.to(device), att.to(device)
    with autocast(device):
        logits = model(input_ids=ids, attention_mask=att).logits
    lp = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
    return torch.stack([lp[i, n_prompt - 1:len(s) - 1].sum() for i, s in enumerate(seqs)]).cpu()


DETOK = [  # undo the Penn-style tokenization of FPB (used for BPC only, applied to both languages)
    (r"``\s*|\s*''", '"'), (r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1"), (r"(?<=\d) \.(?=\d)", "."),
    (r"\s+([,.;:!?%)\]])", r"\1"), (r"([(\[])\s+", r"\1"), (r"\s{2,}", " "),
]


def detok(text):
    for pat, rep in DETOK:
        text = re.sub(pat, rep, text)
    return text.strip()


def test_rows(rows, args):
    test = [r for r in rows if r["split"] == "test"]
    if not args.limit_test:
        return test
    return test[::max(1, len(test) // args.limit_test)][:args.limit_test]  # evenly spaced, smoke tests only


# ---------------------------------------------------------------- task runners

def run_ft(task, m, cfg, rows, subsets, device, args):
    tc = cfg["train"]
    seed = task["seed"]
    set_seed(seed)
    train = [r for r in rows if r["split"] == "train"]
    if task["size"] != "full":
        keep = set(subsets[str(seed)][task["size"]])
        train = [r for r in train if r["id"] in keep]
    test = test_rows(rows, args)

    cuda = device.startswith("cuda")
    tok = load_tok(m["path"])
    # Decoders: bf16 base weights (LoRA adapters are kept in fp32 by peft). Encoders: fp32 weights + bf16 autocast.
    dtype = torch.bfloat16 if cuda and m["kind"] == "decoder" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(m["path"], num_labels=3, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.get_input_embeddings().weight.numel()
    if m["kind"] == "decoder":
        lc = tc["lora"]
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=lc["r"], lora_alpha=lc["alpha"],
                                                 lora_dropout=lc["dropout"], target_modules=lc["target_modules"]))
        lr = tc["lr_decoder"]
    else:
        lr = tc["lr_encoder"]
    model.to(device)

    bs = tc["batch_size"]
    steps = max(tc["epochs"] * math.ceil(len(train) / bs), tc["min_steps"])
    if args.max_steps:
        steps = min(steps, args.max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=tc["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(tc["warmup_ratio"] * steps), steps)
    gen = torch.Generator().manual_seed(seed)
    lang = task["train_lang"]

    model.train()
    step, losses = 0, []
    while step < steps:
        order = torch.randperm(len(train), generator=gen).tolist()
        for i in range(0, len(order), bs):
            if step >= steps:
                break
            batch = [train[j] for j in order[i:i + bs]]
            enc = tok([r[lang] for r in batch], padding=True, truncation=True, max_length=cfg["max_len"],
                      return_tensors="pt").to(device)
            y = torch.tensor([r["label"] for r in batch], device=device)
            with autocast(device):
                loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
            step += 1

    model.eval()
    preds, res = [], {}
    for tl in ("en", "hu"):
        out = []
        with torch.no_grad():
            for i in range(0, len(test), 32):
                enc = tok([r[tl] for r in test[i:i + 32]], padding=True, truncation=True, max_length=cfg["max_len"],
                          return_tensors="pt").to(device)
                with autocast(device):
                    out.append(torch.softmax(model(**enc).logits.float(), -1).cpu())
        probs = torch.cat(out).numpy()
        res[tl] = evaluate(test, probs)
        preds += [{"id": r["id"], "lang": tl, "gold": r["label"], "pred": int(p.argmax()),
                   "probs": [round(float(x), 5) for x in p]} for r, p in zip(test, probs)]
    info = {"n_train": len(train), "steps": steps, "lr": lr, "loss_first": losses[0],
            "loss_last10": float(np.mean(losses[-10:])), "n_params": n_params, "n_embed": n_embed}
    return preds, {"test": res, "train_info": info}


def load_causal(m, device, cpu_bf16=False):
    dtype = torch.bfloat16 if device.startswith("cuda") or cpu_bf16 else torch.float32
    return AutoModelForCausalLM.from_pretrained(m["path"], dtype=dtype).to(device).eval()


def run_c0(task, m, cfg, rows, device, args):
    tok = load_tok(m["path"])
    model = load_causal(m, device, args.cpu_bf16)
    test = test_rows(rows, args)
    pl, tl = task["prompt_lang"], task["test_lang"]
    # Each label is scored as logsumexp over its lowercase and capitalized form.
    variants = [v for w in LABELS[pl] for v in (w, w.capitalize())]

    def scores(text):
        msgs = [{"role": "user", "content": PROMPTS[pl].format(text=text)}]
        # Fixed date_string: the Llama template would otherwise insert today's date (other templates ignore it).
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                    date_string="26 Jul 2024")
        p = tok(s, add_special_tokens=False)["input_ids"]  # the template already contains BOS if any
        # Label tokens are taken from the joint tokenization: sentencepiece tokenizers would otherwise add a
        # word-initial space marker that the model would not produce right after the prompt's newline.
        seqs = [tok(s + v, add_special_tokens=False)["input_ids"] for v in variants]
        assert all(q[:len(p)] == p and len(q) > len(p) for q in seqs), "label merges into the prompt"
        lp = cont_logprob(model, seqs, len(p), device, tok.pad_token_id)
        return torch.logsumexp(lp.view(3, 2), 1).numpy()

    prior = scores(CONTENT_FREE)
    S = np.stack([scores(r[tl]) for r in test])
    probs, probs_cal = softmax(S), softmax(S - prior)
    preds = [{"id": r["id"], "lang": tl, "gold": r["label"], "pred": int(p.argmax()), "pred_cal": int(q.argmax()),
              "probs": [round(float(x), 5) for x in p], "probs_cal": [round(float(x), 5) for x in q]}
             for r, p, q in zip(test, probs, probs_cal)]
    return preds, {"raw": evaluate(test, probs), "calibrated": evaluate(test, probs_cal),
                   "prior_logprob": [float(x) for x in prior]}


def run_bpc(task, m, cfg, rows, device, args):
    tok = load_tok(m["path"])
    model = load_causal(m, device, args.cpu_bf16)
    test = test_rows(rows, args)
    # Same start token for every model, so that every text token is predicted.
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    with open(os.path.join(cfg["data"]["out_dir"], "hu_native.txt"), encoding="utf-8") as f:
        native = f.read().splitlines()
    if args.limit_test:
        native = native[::max(1, len(native) // args.limit_test)][:args.limit_test]
    # en/hu: detokenized FPB test sentences; hu_native: native HU economic news sentences (already untokenized)
    corpora = {"en": [detok(r["en"]) for r in test], "hu": [detok(r["hu"]) for r in test], "hu_native": native}
    res = {}
    for lang, corpus in corpora.items():
        bits = chars = ntok = 0
        for i in range(0, len(corpus), 16):
            texts = corpus[i:i + 16]
            seqs = [[start] + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
            bits += -cont_logprob(model, seqs, 1, device, tok.pad_token_id).sum().item() / math.log(2)
            chars += sum(map(len, texts))
            ntok += sum(len(s) - 1 for s in seqs)
        res[lang] = {"bpc": bits / chars, "tokens": ntok, "chars": chars}
    res["hu_en_bpc"] = res["hu"]["bpc"] / res["en"]["bpc"]
    return [], res


# ---------------------------------------------------------------- main

def short(t, r):
    if t["type"] == "ft":
        return f"f1 en={r['test']['en']['all']['macro_f1']:.3f} hu={r['test']['hu']['all']['macro_f1']:.3f}"
    if t["type"] == "c0":
        return f"f1={r['raw']['all']['macro_f1']:.3f} cal={r['calibrated']['all']['macro_f1']:.3f}"
    return f"bpc en={r['en']['bpc']:.3f} hu={r['hu']['bpc']:.3f} hu_native={r['hu_native']['bpc']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default="cuda:0", help="e.g. cuda:0, cuda:1, cpu")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--only", help="comma list of task types: ft,c0,bpc")
    ap.add_argument("--models", help="comma list of model names from config")
    ap.add_argument("--sizes", help="comma list of ft sizes, e.g. 100,full")
    ap.add_argument("--seeds", help="comma list of ft seeds")
    ap.add_argument("--max_steps", type=int, help="cap training steps (smoke tests only)")
    ap.add_argument("--limit_test", type=int, help="use only N evenly spaced test sentences (smoke tests only)")
    ap.add_argument("--cpu_bf16", action="store_true", help="bf16 weights on CPU for c0/bpc (low-memory smoke tests)")
    ap.add_argument("--status", action="store_true", help="print task counts and exit")
    ap.add_argument("--clear_stale", action="store_true", help="remove RUNNING/ERROR of unfinished tasks and exit")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tasks = build_tasks(cfg, args)

    if args.status or args.clear_stale:
        states = {}
        for t in tasks:
            d = os.path.join(args.out, t["id"])
            s = task_state(d)
            if args.clear_stale and s in ("running", "failed"):
                for f in ("RUNNING", "ERROR.txt"):
                    if os.path.exists(os.path.join(d, f)):
                        os.remove(os.path.join(d, f))
                s = "pending"
            states.setdefault(s, []).append(t["id"])
        print("STATUS " + " ".join(f"{k}={len(states.get(k, []))}" for k in ("done", "running", "failed", "pending"))
              + f" total={len(tasks)}")
        for tid in states.get("failed", []):
            print(f"FAILED {tid}")
        return

    d = cfg["data"]["out_dir"]
    with open(os.path.join(d, "fpb_split.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with open(os.path.join(d, "subsets.json"), encoding="utf-8") as f:
        subsets = json.load(f)
    models = {m["name"]: m for m in cfg["models"]}
    versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    done, failed = [], []
    for t in tasks:
        run_dir = os.path.join(args.out, t["id"])
        if not claim(run_dir):
            continue
        t0 = time.time()
        print(f"START {t['id']} {time.strftime('%H:%M:%S')}", flush=True)
        try:
            m = models[t["model"]]
            if t["type"] == "ft":
                preds, res = run_ft(t, m, cfg, rows, subsets, args.device, args)
            elif t["type"] == "c0":
                preds, res = run_c0(t, m, cfg, rows, args.device, args)
            else:
                preds, res = run_bpc(t, m, cfg, rows, args.device, args)
            with open(os.path.join(run_dir, "preds.jsonl"), "w", encoding="utf-8") as f:
                for p in preds:
                    f.write(json.dumps(p) + "\n")
            meta = {"task": t, "device": args.device, "seconds": round(time.time() - t0, 1), "versions": versions,
                    "limit_test": args.limit_test, "max_steps": args.max_steps}
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({**meta, "results": res}, f, indent=1)
            line = f"DONE {t['id']} {short(t, res)} t={time.time() - t0:.0f}s"
            done.append(line)
        except Exception as e:
            with open(os.path.join(run_dir, "ERROR.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            line = f"FAIL {t['id']} {type(e).__name__}: {str(e)[:100]}"
            failed.append(line)
        finally:
            os.remove(os.path.join(run_dir, "RUNNING"))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(line, flush=True)

    states = [task_state(os.path.join(args.out, t["id"])) for t in tasks]
    print("\n===== SUMMARY (retype this block) =====")
    print(f"RUN device={args.device} done_now={len(done)} failed_now={len(failed)} "
          f"total_done={states.count('done')}/{len(tasks)} failed={states.count('failed')}")
    for line in failed:
        print(line)
    print("=======================================")


if __name__ == "__main__":
    main()
