#!/usr/bin/env python3
"""
Fine-tune microsoft/unixcoder-base for binary code classification (human vs AI).
UniXcoder is pretrained on 6+ languages (Python, Java, JS, Ruby, Go, PHP)
so it should generalize better to unseen languages than CodeBERT.
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================================================================
# Config
# ======================================================================
MODEL_NAME = "microsoft/unixcoder-base"
MAX_LENGTH = 512
BATCH_SIZE = 16          # small for 10GB cgroup limit
GRAD_ACCUM = 4           # effective batch = 64
LR = 2e-5
EPOCHS = 2               # diminishing returns after epoch 2
WARMUP_RATIO = 0.10
NUM_WORKERS = 2
SEED = 42

DATA_DIR = Path("Task_A")
SAVE_DIR = Path("models/unixcoder"); SAVE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("FINE-TUNE UniXcoder — Binary Code Classification (Human vs AI)")
print("=" * 70)
print(f"  Model:     {MODEL_NAME}")
print(f"  Device:    {device}")
if torch.cuda.is_available():
    print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    print(f"  GPU Mem:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Batch:     {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective")
print(f"  LR:        {LR}")
print(f"  Epochs:    {EPOCHS}")
print(f"  MaxLen:    {MAX_LENGTH}")
print("=" * 70, flush=True)


# ======================================================================
# Dataset
# ======================================================================
class CodeDataset(Dataset):
    """Tokenizes on-the-fly to minimize memory footprint."""
    def __init__(self, codes, labels=None, tokenizer=None, max_length=512):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = str(self.codes[idx]) if self.codes[idx] is not None else ""
        enc = self.tokenizer(code, max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ======================================================================
# Inference helpers
# ======================================================================
@torch.no_grad()
def evaluate_model(model, loader, desc="Eval"):
    """Returns (preds, labels_or_None, avg_loss, probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0; n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        with autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.append(preds)
        all_probs.append(probs)
        if labels is not None:
            total_loss += out.loss.item(); n += 1
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels) if all_labels else None
    return all_preds, all_labels, total_loss / max(n, 1), all_probs


@torch.no_grad()
def predict_probs(model, loader):
    """Returns probabilities only (for test set without labels)."""
    model.eval()
    all_probs = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, attention_mask=mask)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ======================================================================
# [1/5] Load data
# ======================================================================
print("\n" + "=" * 70)
print("[1/5] LOADING DATA")
print("=" * 70, flush=True)
t0 = time.time()

train_df = pd.read_parquet(DATA_DIR / "train.parquet", columns=["code", "label"])
val_df = pd.read_parquet(DATA_DIR / "validation.parquet", columns=["code", "label"])
test_sample_df = pd.read_parquet(DATA_DIR / "test_sample.parquet")

print(f"  Train:       {len(train_df):,} rows")
print(f"  Validation:  {len(val_df):,} rows")
print(f"  Test sample: {len(test_sample_df):,} rows")
print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

# ======================================================================
# [2/5] Load tokenizer & model
# ======================================================================
print("\n" + "=" * 70)
print("[2/5] LOADING MODEL & TOKENIZER")
print("=" * 70, flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.gradient_checkpointing_enable()
model = model.to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"  Parameters:  {param_count:,}")
print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

# Build datasets & loaders
train_dataset = CodeDataset(train_df["code"].values, train_df["label"].values, tokenizer, MAX_LENGTH)
val_dataset = CodeDataset(val_df["code"].values, val_df["label"].values, tokenizer, MAX_LENGTH)
ts_dataset = CodeDataset(test_sample_df["code"].values, test_sample_df["label"].values, tokenizer, MAX_LENGTH)

del train_df, val_df; gc.collect()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
ts_loader = DataLoader(ts_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

# ======================================================================
# [3/5] Training setup
# ======================================================================
print("\n" + "=" * 70)
print("[3/5] TRAINING")
print("=" * 70, flush=True)

total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
eval_steps = len(train_loader) // (GRAD_ACCUM * 2)  # eval ~2x per epoch (every 0.5 epoch)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

print(f"  Steps/epoch:    {len(train_loader)}")
print(f"  Total optim:    {total_steps}")
print(f"  Warmup steps:   {warmup_steps}")
print(f"  Eval every:     {eval_steps} optim steps (~0.5 epoch)")
print(flush=True)

best_val_f1 = 0.0
global_step = 0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0; batch_count = 0
    optimizer.zero_grad()

    print(f"\n--- Epoch {epoch}/{EPOCHS} ---", flush=True)

    for step, batch in enumerate(train_loader, 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        epoch_loss += out.loss.item()
        batch_count += 1

        if step % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        if step % 500 == 0:
            avg = epoch_loss / batch_count
            elapsed = time.time() - t_start
            print(f"  Step {step}/{len(train_loader)} | loss={avg:.4f} | time={elapsed:.0f}s", flush=True)

        # Mid-epoch eval (every 0.5 epoch)
        if step % (eval_steps * GRAD_ACCUM) == 0:
            preds, labels_arr, val_loss, _ = evaluate_model(model, val_loader)
            val_f1 = f1_score(labels_arr, preds)
            val_acc = accuracy_score(labels_arr, preds)
            elapsed = time.time() - t_start
            print(f"  ** EVAL step {global_step}: Val F1={val_f1:.4f} Acc={val_acc:.4f} Loss={val_loss:.4f} | {elapsed:.0f}s", flush=True)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(SAVE_DIR / "best")
                tokenizer.save_pretrained(SAVE_DIR / "best")
                print(f"     -> New best model saved! F1={val_f1:.4f}", flush=True)
            model.train()

    avg_loss = epoch_loss / batch_count
    elapsed = time.time() - t_start
    print(f"\n  Epoch {epoch} complete | Avg loss: {avg_loss:.4f} | Elapsed: {elapsed:.0f}s", flush=True)

    # End-of-epoch eval
    preds, labels_arr, val_loss, _ = evaluate_model(model, val_loader)
    val_f1 = f1_score(labels_arr, preds)
    val_acc = accuracy_score(labels_arr, preds)
    print(f"  End-of-epoch eval: Val F1={val_f1:.4f} Acc={val_acc:.4f} Loss={val_loss:.4f}", flush=True)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_pretrained(SAVE_DIR / "best")
        tokenizer.save_pretrained(SAVE_DIR / "best")
        print(f"     -> New best model saved! F1={val_f1:.4f}", flush=True)

total_train_time = time.time() - t_start
print(f"\n  Training done. Best Val F1: {best_val_f1:.4f}")
print(f"  Total training time: {total_train_time:.0f}s ({total_train_time/60:.1f}min)")

# ======================================================================
# [4/5] Final evaluation (reload best model fresh to avoid OOM)
# ======================================================================
print("\n" + "=" * 70)
print("[4/5] FINAL EVALUATION")
print("=" * 70, flush=True)

# Free training memory
del model, optimizer, scheduler, scaler
gc.collect()
torch.cuda.empty_cache()

# Reload best model
print("  Reloading best model...", flush=True)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR / "best")
model = model.to(device)
model.eval()

# --- Validation ---
print("\n--- Validation Set ---", flush=True)
t0 = time.time()
val_preds, val_labels, val_loss, val_probs = evaluate_model(model, val_loader, "Val")
val_f1 = f1_score(val_labels, val_preds)
print(f"  F1:   {val_f1:.4f}")
print(f"  Acc:  {accuracy_score(val_labels, val_preds):.4f}")
print(f"  Loss: {val_loss:.4f}")
print(classification_report(val_labels, val_preds, target_names=["Human", "AI"], digits=4))
print(f"  Eval time: {time.time() - t0:.1f}s", flush=True)

# Save val probs
np.save(CACHE_DIR / "unixcoder_val_probs.npy", val_probs)
print(f"  Saved val probs -> {CACHE_DIR / 'unixcoder_val_probs.npy'}")
del val_preds, val_labels, val_probs; gc.collect()

# --- Test sample ---
print("\n--- Test Sample (with unseen languages) ---", flush=True)
t0 = time.time()
ts_preds, ts_labels, ts_loss, ts_probs = evaluate_model(model, ts_loader, "TS")
ts_f1 = f1_score(ts_labels, ts_preds)
print(f"  F1:   {ts_f1:.4f}")
print(f"  Acc:  {accuracy_score(ts_labels, ts_preds):.4f}")
print(classification_report(ts_labels, ts_preds, target_names=["Human", "AI"], digits=4))

# Per-language analysis
print("  Per-language breakdown:")
print(f"  {'Language':<12s} {'Count':>6s} {'F1':>7s} {'Acc':>7s} {'AI%':>6s}")
print(f"  {'-'*40}")
for lang in sorted(test_sample_df["language"].unique()):
    mask = test_sample_df["language"].values == lang
    y_t = ts_labels[mask]; y_p = ts_preds[mask]
    lang_f1 = f1_score(y_t, y_p, zero_division=0)
    lang_acc = accuracy_score(y_t, y_p)
    ai_pct = y_t.mean() * 100
    print(f"  {lang:<12s} {mask.sum():>6d} {lang_f1:>7.3f} {lang_acc:>7.3f} {ai_pct:>5.1f}%")
print(f"  Eval time: {time.time() - t0:.1f}s", flush=True)

del ts_preds, ts_labels, ts_probs; gc.collect()

# ======================================================================
# [5/5] Full test set prediction
# ======================================================================
print("\n" + "=" * 70)
print("[5/5] FULL TEST SET PREDICTION")
print("=" * 70, flush=True)
t0 = time.time()

test_df = pd.read_parquet(DATA_DIR / "test.parquet")
print(f"  Test rows: {len(test_df):,}", flush=True)

test_dataset = CodeDataset(test_df["code"].values, None, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

test_probs = predict_probs(model, test_loader)
test_preds = (test_probs >= 0.5).astype(int)

# Save submission
submission = pd.DataFrame({"ID": test_df["ID"].values, "label": test_preds})
submission.to_csv(DATA_DIR / "submission_unixcoder.csv", index=False)
print(f"  Submission saved -> {DATA_DIR / 'submission_unixcoder.csv'}")
print(f"  Predictions: {test_preds.sum():,} AI / {(1 - test_preds).sum():,} Human")
print(f"  Mean prob: {test_probs.mean():.4f}")

# Save probs for ensembling
np.save(CACHE_DIR / "unixcoder_test_probs.npy", test_probs)
print(f"  Test probs saved -> {CACHE_DIR / 'unixcoder_test_probs.npy'}")

print(f"  Prediction time: {time.time() - t0:.1f}s", flush=True)

# ======================================================================
# Done
# ======================================================================
total_time = time.time() - t_start
print("\n" + "=" * 70)
print("COMPLETE")
print(f"  Best Val F1:      {best_val_f1:.4f}")
print(f"  Test Sample F1:   {ts_f1:.4f}")
print(f"  Total time:       {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"  Submission:       {DATA_DIR / 'submission_unixcoder.csv'}")
print(f"  Val probs:        {CACHE_DIR / 'unixcoder_val_probs.npy'}")
print(f"  Test probs:       {CACHE_DIR / 'unixcoder_test_probs.npy'}")
print(f"  Best model:       {SAVE_DIR / 'best'}")
print("=" * 70)
