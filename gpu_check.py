#!/usr/bin/env python
"""Offline GPU and training-chain check for the air-gapped server.

Checks, in order:
  1. Environment: library versions, offline flags, visible GPUs.
  2. Per device: load the model as a 3-class sequence classifier, wrap it
     with LoRA, run a short training loop, and run inference.
  3. Bits-per-character (BPC) with the same weights loaded as a causal LM.
  4. Write access to the output directory.

Usage (inside the container):
  python gpu_check.py --model /path/to/Qwen3-0.6B
  python gpu_check.py --model /path/to/Qwen3-0.6B --devices cpu   # local test

The script prints a short summary block at the end. Retype only that block.
"""
import os

# Must be set before transformers / huggingface_hub are imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import math
import time
import traceback

import torch

# A tiny parallel EN/HU sample with all three classes (0=negative, 1=neutral, 2=positive).
SAMPLES = [
    ("Operating profit fell sharply compared to the previous year.", 0),
    ("The company will publish its annual report in March.", 1),
    ("Net sales increased by 12 percent in the third quarter.", 2),
    ("Az üzemi eredmény jelentősen csökkent az előző évhez képest.", 0),
    ("A vállalat márciusban teszi közzé az éves jelentését.", 1),
    ("A nettó árbevétel 12 százalékkal nőtt a harmadik negyedévben.", 2),
]
BPC_TEXT = {
    "en": "The central bank kept its base rate unchanged, and inflation is expected to slow next year.",
    "hu": "A jegybank változatlanul hagyta az alapkamatot, és az infláció lassulása várható jövőre.",
}


def check_env():
    import transformers, peft, accelerate, sklearn
    info = {
        "python_torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "sklearn": sklearn.__version__,
        "hf_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    return info


def check_train(model_path, device, steps, max_len):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(0)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3, dtype=dtype
    )
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    texts = [s for s, _ in SAMPLES]
    labels = torch.tensor([y for _, y in SAMPLES], device=device)
    batch = tok(texts, padding=True, truncation=True, max_length=max_len,
                return_tensors="pt").to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    losses = []
    for _ in range(steps):
        out = model(**batch, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(**batch).logits.float(), dim=-1)
    peak_gb = (torch.cuda.max_memory_allocated(device) / 1e9) if device.startswith("cuda") else 0.0

    ok = all(math.isfinite(l) for l in losses) and losses[-1] < losses[0] \
        and torch.allclose(probs.sum(-1), torch.ones(len(texts), device=device), atol=1e-3)
    res = {
        "ok": bool(ok),
        "loss_first": round(losses[0], 4),
        "loss_last": round(losses[-1], 4),
        "train_acc": round((probs.argmax(-1) == labels).float().mean().item(), 3),
        "trainable_params": n_train,
        "peak_mem_gb": round(peak_gb, 2),
        "seconds": round(time.time() - t0, 1),
    }
    del model, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def check_bpc(model_path, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    res = {}
    for lang, text in BPC_TEXT.items():
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.float().item()  # mean NLL in nats over predicted tokens
        n_pred = ids.shape[1] - 1
        bpc = loss * n_pred / math.log(2) / len(text)
        res[lang] = {"bpc": round(bpc, 3), "tokens": ids.shape[1], "chars": len(text)}
    res["ok"] = all(math.isfinite(res[l]["bpc"]) and res[l]["bpc"] > 0 for l in BPC_TEXT)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Local path to the HF snapshot")
    ap.add_argument("--devices", default="all", help="'all' (every GPU), 'cpu', or e.g. 'cuda:0'")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--out", default="gpu_check_out")
    args = ap.parse_args()

    # Fail early with a clear message instead of a misleading "couldn't connect to huggingface.co".
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        print(f"MODEL PATH FAIL: no config.json in '{os.path.abspath(args.model)}' (cwd: {os.getcwd()})")
        raise SystemExit(1)

    report = {"model": args.model}
    summary = []

    try:
        env = check_env()
        report["env"] = env
        summary.append(f"ENV torch={env['python_torch']} tf={env['transformers']} "
                       f"peft={env['peft']} gpus={env['gpu_count']} offline={env['hf_offline']}")
    except Exception as e:
        report["env_error"] = traceback.format_exc()
        summary.append(f"ENV FAIL {type(e).__name__}: {e}")

    if args.devices == "all":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    else:
        devices = [d.strip() for d in args.devices.split(",")]

    for dev in devices:
        try:
            r = check_train(args.model, dev, args.steps, args.max_len)
            report[f"train_{dev}"] = r
            summary.append(f"TRAIN {dev} {'OK' if r['ok'] else 'FAIL'} "
                           f"loss {r['loss_first']}->{r['loss_last']} mem={r['peak_mem_gb']}GB t={r['seconds']}s")
        except Exception as e:
            report[f"train_{dev}_error"] = traceback.format_exc()
            summary.append(f"TRAIN {dev} FAIL {type(e).__name__}: {str(e)[:120]}")

    try:
        r = check_bpc(args.model, devices[0])
        report["bpc"] = r
        summary.append(f"BPC {'OK' if r['ok'] else 'FAIL'} en={r['en']['bpc']} hu={r['hu']['bpc']} "
                       f"tok en/hu={r['en']['tokens']}/{r['hu']['tokens']}")
    except Exception as e:
        report["bpc_error"] = traceback.format_exc()
        summary.append(f"BPC FAIL {type(e).__name__}: {str(e)[:120]}")

    try:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "gpu_check.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        summary.append(f"WRITE OK {path}")
    except Exception as e:
        summary.append(f"WRITE FAIL {type(e).__name__}: {e}")

    print("\n===== SUMMARY (retype this block) =====")
    for line in summary:
        print(line)
    print("=======================================")


if __name__ == "__main__":
    main()
