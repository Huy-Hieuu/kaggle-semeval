#!/usr/bin/env python3
"""
Phase 3: Fine-tune CodeBERT — optimized for 10GB memory limit.
Uses gradient checkpointing and smaller batch to fit in memory.
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

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 16  # small for 10GB limit
GRAD_ACCUM = 4   # effective batch = 64
LR = 2e-5
EPOCHS = 3
WARMUP_RATIO = 0.10
SEED = 42

DATA_DIR = Path("Task_A")
SAVE_DIR = Path("models/codebert"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PHASE 3 — Fine-tune CodeBERT")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE*GRAD_ACCUM} effective")
print("=" * 70, flush=True)


class CodeDataset(Dataset):
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
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ===== Load data =====
print("\n[1/4] Loading data...", flush=True)
train_df = pd.read_parquet(DATA_DIR / "train.parquet", columns=["code", "label"])
val_df = pd.read_parquet(DATA_DIR / "validation.parquet", columns=["code", "label"])
test_sample_df = pd.read_parquet(DATA_DIR / "test_sample.parquet")
# test loaded later to save memory

print(f"  Train: {len(train_df)}, Val: {len(val_df)}, TS: {len(test_sample_df)}")

# ===== Tokenizer & model =====
print("\n[2/4] Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.gradient_checkpointing_enable()  # save memory
model = model.to(device)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# Datasets
train_dataset = CodeDataset(train_df["code"].values, train_df["label"].values, tokenizer, MAX_LENGTH)
val_dataset = CodeDataset(val_df["code"].values, val_df["label"].values, tokenizer, MAX_LENGTH)
ts_dataset = CodeDataset(test_sample_df["code"].values, test_sample_df["label"].values, tokenizer, MAX_LENGTH)

del train_df, val_df; gc.collect()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
ts_loader = DataLoader(ts_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

# ===== Training setup =====
total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
eval_steps = len(train_loader) // (GRAD_ACCUM * 2)  # eval ~2x per epoch

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

print(f"  Steps/epoch: {len(train_loader)}, Total steps: {total_steps}, Eval every: {eval_steps}")

# ===== Eval function =====
@torch.no_grad()
def evaluate_model(loader, desc="Eval"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0; n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        with autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
        preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
        all_preds.append(preds)
        if labels is not None:
            total_loss += out.loss.item(); n += 1
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels) if all_labels else None
    return all_preds, all_labels, total_loss / max(n, 1)


# ===== Training =====
print("\n[3/4] Training...", flush=True)
best_val_f1 = 0.0
global_step = 0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0; batch_count = 0
    optimizer.zero_grad()

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
            print(f"  Epoch {epoch} step {step}/{len(train_loader)} loss={avg:.4f}", flush=True)

        # Mid-epoch eval
        if step % (eval_steps * GRAD_ACCUM) == 0:
            preds, labels_arr, val_loss = evaluate_model(val_loader)
            val_f1 = f1_score(labels_arr, preds)
            print(f"  ** Eval step {global_step}: Val F1={val_f1:.4f} Loss={val_loss:.4f}", flush=True)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(SAVE_DIR / "best")
                tokenizer.save_pretrained(SAVE_DIR / "best")
                print(f"     New best! Saved.", flush=True)
            model.train()

    avg_loss = epoch_loss / batch_count
    print(f"\n  Epoch {epoch} done | Avg loss: {avg_loss:.4f} | Time: {time.time()-t_start:.0f}s", flush=True)

print(f"\n  Training done. Best Val F1: {best_val_f1:.4f}")

# ===== Final eval =====
print("\n[4/4] Final evaluation...", flush=True)
# Load best
del model; gc.collect(); torch.cuda.empty_cache()
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR / "best")
model = model.to(device)

# Val
preds, labels_arr, _ = evaluate_model(val_loader)
print(f"\n--- Validation ---")
print(f"  F1: {f1_score(labels_arr, preds):.4f}")
print(classification_report(labels_arr, preds, target_names=["Human", "AI"], digits=4))

# Test sample
ts_preds, ts_labels, _ = evaluate_model(ts_loader)
print(f"--- Test Sample ---")
print(f"  F1: {f1_score(ts_labels, ts_preds):.4f}")
print(classification_report(ts_labels, ts_preds, target_names=["Human", "AI"], digits=4))

# Per-language
print(f"\nPer-language on test_sample:")
for lang in sorted(test_sample_df["language"].unique()):
    mask = test_sample_df["language"].values == lang
    y_t = ts_labels[mask]; y_p = ts_preds[mask]
    f1 = f1_score(y_t, y_p, zero_division=0)
    print(f"  {lang:<12s} F1={f1:.3f} Acc={accuracy_score(y_t, y_p):.3f}")

# Test predictions (chunked)
print(f"\nPredicting on full test set...", flush=True)
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
test_dataset = CodeDataset(test_df["code"].values, None, tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

all_test_preds = []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with autocast("cuda"):
            out = model(input_ids=ids, attention_mask=mask)
        preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
        all_test_preds.append(preds)

test_preds = np.concatenate(all_test_preds)
pd.DataFrame({"ID": test_df["ID"].values, "label": test_preds.astype(int)}).to_csv(
    DATA_DIR / "submission_codebert.csv", index=False
)
model.save_pretrained(SAVE_DIR / "final")
tokenizer.save_pretrained(SAVE_DIR / "final")
print(f"\nDone! Submission saved. Best Val F1: {best_val_f1:.4f}")
