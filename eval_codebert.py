#!/usr/bin/env python3
"""Evaluate saved CodeBERT and generate predictions (lightweight, no training)."""
import gc, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = Path("Task_A")
SAVE_DIR = Path("models/codebert/best")
BATCH_SIZE = 32
MAX_LENGTH = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CodeDataset(Dataset):
    def __init__(self, codes, labels=None, tokenizer=None, max_length=512):
        self.codes = codes; self.labels = labels
        self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.codes)
    def __getitem__(self, idx):
        code = str(self.codes[idx]) if self.codes[idx] is not None else ""
        enc = self.tokenizer(code, max_length=self.max_length, padding="max_length",
                            truncation=True, return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

print("Loading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model = model.to(device).eval()

@torch.no_grad()
def predict_dataset(codes, labels=None, desc=""):
    ds = CodeDataset(codes, labels, tokenizer, MAX_LENGTH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    all_preds, all_probs = [], []
    for batch in dl:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with autocast("cuda"):
            out = model(input_ids=ids, attention_mask=mask)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.append(preds); all_probs.append(probs)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    if labels is not None:
        print(f"\n--- {desc} ---")
        print(f"  Accuracy: {accuracy_score(labels, all_preds):.4f}")
        print(f"  F1: {f1_score(labels, all_preds):.4f}")
        print(classification_report(labels, all_preds, target_names=["Human", "AI"], digits=4))
    return all_preds, all_probs

# === Validation ===
print("\nEvaluating validation set...", flush=True)
val_df = pd.read_parquet(DATA_DIR/"validation.parquet", columns=["code", "label"])
val_preds, val_probs = predict_dataset(val_df["code"].values, val_df["label"].values, "Validation")
del val_df; gc.collect()

# === Test Sample ===
print("Evaluating test_sample...", flush=True)
ts_df = pd.read_parquet(DATA_DIR/"test_sample.parquet")
ts_preds, ts_probs = predict_dataset(ts_df["code"].values, ts_df["label"].values, "Test Sample")

# Per-language
print("\nPer-language on test_sample:")
for lang in sorted(ts_df["language"].unique()):
    mask = ts_df["language"].values == lang
    y_t = ts_df["label"].values[mask]; y_p = ts_preds[mask]
    f1 = f1_score(y_t, y_p, zero_division=0)
    acc = accuracy_score(y_t, y_p)
    h_r = ((y_p==0)&(y_t==0)).sum() / max((y_t==0).sum(), 1)
    a_r = ((y_p==1)&(y_t==1)).sum() / max((y_t==1).sum(), 1)
    print(f"  {lang:<12s} N={mask.sum():>4d} Acc={acc:.3f} F1={f1:.3f} H_Recall={h_r:.3f} AI_Recall={a_r:.3f}")
del ts_df; gc.collect()

# === Full Test ===
print("\nPredicting full test set...", flush=True)
test_df = pd.read_parquet(DATA_DIR/"test.parquet")
test_preds, test_probs = predict_dataset(test_df["code"].values, desc="Test")

pd.DataFrame({"ID": test_df["ID"].values, "label": test_preds.astype(int)}).to_csv(
    DATA_DIR/"submission_codebert.csv", index=False
)
# Also save probabilities for ensembling later
np.save("cache/codebert_test_probs.npy", test_probs)
np.save("cache/codebert_val_probs.npy", val_probs)
print(f"\nSaved submission_codebert.csv and probability caches")
print(f"Test label distribution: {np.unique(test_preds, return_counts=True)}")
