#!/usr/bin/env python3
"""
Phase 4: Domain Shift Handling (10GB memory limit)
Key insight from analysis:
- test_sample is 77.7% human but models predict almost all unseen-language code as AI
- has_tabs (corr=0.80) is Python-specific and doesn't transfer
- Need features that capture AI vs human patterns ACROSS languages

Strategy:
1. Agnostic features model (already done, TS F1=0.39)
2. Use HashingVectorizer batch approach for char TF-IDF
3. Combined model with agnostic features + SVD
4. Generate submission
"""
import gc, time, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb

warnings.filterwarnings("ignore")
DATA_DIR = Path("Task_A")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(exist_ok=True)
MAX_CODE_LEN = 500

print("=" * 70)
print("PHASE 4 — Domain Shift: Agnostic Features + Char SVD")
print("=" * 70, flush=True)

from features import extract_features

# Language-agnostic features (no Python-specific ones)
AGNOSTIC = [
    'code_len', 'line_count', 'avg_line_len', 'max_line_len',
    'empty_line_ratio', 'whitespace_ratio', 'digit_ratio',
    'char_entropy', 'unique_char_ratio', 'repeated_line_ratio',
    'token_diversity', 'avg_token_len',
    'indent_consistency', 'trailing_whitespace_ratio', 'bracket_ratio',
    'max_nesting_depth', 'avg_tokens_per_line',
    'line_len_std', 'line_len_skew', 'blank_line_cluster_count',
    'alpha_ratio', 'upper_ratio', 'lower_ratio', 'punct_ratio',
    'paren_ratio', 'brace_ratio', 'square_bracket_ratio',
    'operator_ratio', 'string_literal_ratio',
]

def get_agnostic_cached(split):
    """Get agnostic features from full cached features."""
    full = np.load(CACHE_DIR / f"hc_{split}.npy")
    # Get column indices matching AGNOSTIC names
    _tmp = pd.read_parquet(DATA_DIR/"train.parquet", columns=["code"]).head(1)
    all_names = list(extract_features(_tmp).columns)
    idx = [all_names.index(f) for f in AGNOSTIC]
    return full[:, idx]

# ===== Load agnostic features =====
print("\n[1/5] Loading agnostic features...", flush=True)
X_ag_train = get_agnostic_cached("train")
X_ag_val = get_agnostic_cached("validation")
X_ag_ts = get_agnostic_cached("test_sample")
y_train = pd.read_parquet(DATA_DIR/"train.parquet", columns=["label"])["label"].values
y_val = pd.read_parquet(DATA_DIR/"validation.parquet", columns=["label"])["label"].values
y_ts = pd.read_parquet(DATA_DIR/"test_sample.parquet", columns=["label"])["label"].values
print(f"  Train agnostic: {X_ag_train.shape}", flush=True)

# ===== Char TF-IDF SVD (reuse cached if available) =====
print("\n[2/5] Char TF-IDF + SVD...", flush=True)

hv = HashingVectorizer(
    analyzer="char_wb", ngram_range=(2, 3), n_features=2**16,
    alternate_sign=False, dtype=np.float32,
)

svd_cache = CACHE_DIR / "svd_train.npy"
if svd_cache.exists():
    print("  Loading cached SVD...", flush=True)
    X_svd_train = np.load(svd_cache)
    pipeline = joblib.load(CACHE_DIR / "tfidf_svd_pipeline.joblib")
    tfidf_transformer = pipeline["tfidf"]
    svd = pipeline["svd"]
else:
    # Build from scratch using batch approach
    t0 = time.time()
    train_code = pd.read_parquet(DATA_DIR/"train.parquet", columns=["code"])["code"].values
    import scipy.sparse as sp
    BATCH = 50000
    hash_batches = []
    for i in range(0, len(train_code), BATCH):
        batch = [s[:MAX_CODE_LEN] if s else "" for s in train_code[i:i+BATCH]]
        hash_batches.append(hv.transform(batch))
        del batch; gc.collect()
    del train_code; gc.collect()
    X_hash_train = sp.vstack(hash_batches, format="csr")
    del hash_batches; gc.collect()

    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    X_tfidf_train = tfidf_transformer.fit_transform(X_hash_train)
    del X_hash_train; gc.collect()

    svd = TruncatedSVD(n_components=80, random_state=42)
    X_svd_train = svd.fit_transform(X_tfidf_train).astype(np.float32)
    del X_tfidf_train; gc.collect()

    np.save(svd_cache, X_svd_train)
    joblib.dump({"tfidf": tfidf_transformer, "svd": svd}, CACHE_DIR / "tfidf_svd_pipeline.joblib")
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

def code_to_svd(code_values):
    code_t = [s[:MAX_CODE_LEN] if s else "" for s in code_values]
    X_hash = hv.transform(code_t)
    X_tfidf = tfidf_transformer.transform(X_hash)
    result = svd.transform(X_tfidf).astype(np.float32)
    del X_hash, X_tfidf, code_t; gc.collect()
    return result

val_code = pd.read_parquet(DATA_DIR/"validation.parquet", columns=["code"])["code"].values
X_svd_val = code_to_svd(val_code); del val_code; gc.collect()
ts_code = pd.read_parquet(DATA_DIR/"test_sample.parquet", columns=["code"])["code"].values
X_svd_ts = code_to_svd(ts_code); del ts_code; gc.collect()

# ===== Combine & Train =====
print("\n[3/5] Combining agnostic + SVD and training...", flush=True)
X_train = np.hstack([X_svd_train, X_ag_train]); del X_svd_train, X_ag_train
X_val = np.hstack([X_svd_val, X_ag_val]); del X_svd_val, X_ag_val
X_ts = np.hstack([X_svd_ts, X_ag_ts]); del X_svd_ts, X_ag_ts
gc.collect()
feat_names = [f"svd_{i}" for i in range(80)] + AGNOSTIC
print(f"  Combined: {X_train.shape}", flush=True)

t0 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.03, num_leaves=127,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=50,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
)
del X_train; gc.collect()
print(f"  Best iter: {model.best_iteration_} | Time: {time.time()-t0:.1f}s", flush=True)

# ===== Evaluate =====
print("\n[4/5] Evaluation...", flush=True)
val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
print(f"\n--- Validation ---")
print(f"  F1: {f1_score(y_val, val_pred):.4f}")
print(classification_report(y_val, val_pred, target_names=["Human", "AI"], digits=4))

ts_pred = model.predict(X_ts)
ts_prob = model.predict_proba(X_ts)[:, 1]
print(f"--- Test Sample ---")
print(f"  F1: {f1_score(y_ts, ts_pred):.4f}")
print(classification_report(y_ts, ts_pred, target_names=["Human", "AI"], digits=4))

# Per-language analysis
test_sample = pd.read_parquet(DATA_DIR / "test_sample.parquet")
print(f"\nPer-language performance:")
print(f"  {'Language':<12s} {'N':>5s} {'Acc':>6s} {'F1':>6s} {'H_Recall':>9s} {'AI_Recall':>10s}")
print("-" * 48)
for lang in sorted(test_sample["language"].unique()):
    mask = test_sample["language"].values == lang
    y_true = y_ts[mask]; y_pred = ts_pred[mask]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    h_r = ((y_pred==0)&(y_true==0)).sum() / max((y_true==0).sum(), 1)
    a_r = ((y_pred==1)&(y_true==1)).sum() / max((y_true==1).sum(), 1)
    print(f"  {lang:<12s} {mask.sum():>5d} {acc:>6.3f} {f1:>6.3f} {h_r:>9.3f} {a_r:>10.3f}")

# Threshold analysis — maybe shifting threshold helps for domain shift
print(f"\nThreshold analysis on test_sample:")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    pred_t = (ts_prob >= thresh).astype(int)
    f1 = f1_score(y_ts, pred_t, zero_division=0)
    acc = accuracy_score(y_ts, pred_t)
    print(f"  threshold={thresh:.1f}: Acc={acc:.4f} F1={f1:.4f}")

del X_val, X_ts; gc.collect()

# ===== Test predictions =====
print("\n[5/5] Predicting test...", flush=True)
test = pd.read_parquet(DATA_DIR/"test.parquet")
test_ids = test["ID"].values

CHUNK = 25000
all_preds = []
for i in range(0, len(test), CHUNK):
    chunk = test.iloc[i:i+CHUNK]
    X_hc = np.asarray(extract_features(chunk)[AGNOSTIC], dtype=np.float32)
    X_hc[~np.isfinite(X_hc)] = 0.0
    X_svd = code_to_svd(chunk["code"].values)
    preds = model.predict(np.hstack([X_svd, X_hc]))
    all_preds.append(preds)
    del X_hc, X_svd; gc.collect()
    print(f"  Chunk {i//CHUNK+1}/{(len(test)-1)//CHUNK+1}", flush=True)

test_pred = np.concatenate(all_preds)
pd.DataFrame({"ID": test_ids, "label": test_pred.astype(int)}).to_csv(
    DATA_DIR / "submission_phase4_combined.csv", index=False
)
joblib.dump({"model": model, "hv": hv, "tfidf_transformer": tfidf_transformer,
             "svd": svd, "agnostic_features": AGNOSTIC},
            MODEL_DIR / "phase4_domain.joblib")
print("\nDone! Saved submission + model.")
