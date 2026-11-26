# Tokenization and Language Competence of Small Language Models for Financial Sentiment

Code for the experiments comparing small language models (0.6–4B parameters) on English and Hungarian
financial sentiment classification, on the Financial PhraseBank (FPB) and its human-reviewed Hungarian
translation.

The study asks three questions:

- **H1:** Do small models perform worse on Hungarian than on English on the same sentences?
- **H2:** Is the gap related to Hungarian tokenizer quality (fertility) or to the model's Hungarian
  language competence (bits per character)?
- **H3:** How many more Hungarian than English training sentences are needed for the same performance?

## Data

**Upload in progress.** The Hungarian FPB translation (CC BY-NC-SA 3.0, the exact version used in the
experiments) and the train/test split will be added here.

The native Hungarian news sentences used only for the bits-per-character measure are not distributed.

## Pipeline

All settings (models, seeds, hyperparameters, data paths) are in `config.yaml`.

| Script | Purpose |
|---|---|
| `prepare_data.py` | Deduplication, fixed-seed stratified train/test split, nested learning-curve subsets |
| `tokstats.py` | Tokenizer statistics per model: fertility, share of split words, HU/EN token ratio |
| `gpu_check.py` | Offline check of model loading, a short LoRA training run, and BPC on each GPU |
| `run.py` | All experiments: fine-tuning (incl. learning curves), zero-shot classification, BPC. Resumable; several workers (one per GPU) share the task list |
| `make_report.py` | Self-contained HTML report, summary tables, bootstrap CIs and correlations from the run outputs |

Typical order:

```bash
pip install -r requirements.txt
python prepare_data.py
python tokstats.py
nohup python run.py --device cuda:0 > worker0.log 2>&1 &
nohup python run.py --device cuda:1 > worker1.log 2>&1 &
python run.py --status
python make_report.py
```

Models are loaded from local folders (`path:` in `config.yaml`); the scripts run offline.

## Setup in brief

- Decoders: instruction-tuned checkpoints fine-tuned as sequence classifiers with LoRA
  (r=16, alpha=32, lr 2e-4); encoders (huBERT, XLM-R base): full fine-tuning (lr 2e-5).
  Identical hyperparameters for all models, no per-model tuning, no validation set.
- Training length: max(3 epochs, 150 steps); three seeds, which also select the training subsets.
- Learning curves: 100 / 250 / 500 / 1000 / full training set (3,866 sentences); test set: 967 sentences.
- Zero-shot: label log-probability with the model's chat template, prompt in the test language.

## License

Code: MIT (see `LICENSE`). The FPB data and its translation are licensed under CC BY-NC-SA 3.0.
