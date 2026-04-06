#!/usr/bin/env python3
"""Phase 3: TF-IDF + LightGBM — batch processing for 10GB memory limit."""
import gc, time, warnings, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
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
MAX_CODE_LEN = 500  # aggressive truncation for memory

print("=" * 70)
print("PHASE 3 — TF-IDF + LightGBM (10GB limit, batch mode)")
print("=" * 70, flush=True)

from features import extract_features

# ===== Cached features =====
def load_cached_features(split_name):
    cache_path = CACHE_DIR / f"hc_{split_name}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    df = pd.read_parquet(DATA_DIR / f"{split_name}.parquet")
    feats = np.asarray(extract_features(df), dtype=np.float32)
    feats[~np.isfinite(feats)] = 0.0
    np.save(cache_path, feats)
    del df; gc.collect()
    return feats

print("\n[1/5] Loading cached features...", flush=True)
X_hc_train = load_cached_features("train")
X_hc_val = load_cached_features("validation")
X_hc_ts = load_cached_features("test_sample")

_tmp = pd.read_parquet(DATA_DIR/"train.parquet", columns=["code"]).head(2)
hc_names = list(extract_features(_tmp).columns)
del _tmp; gc.collect()

y_train = pd.read_parquet(DATA_DIR/"train.parquet", columns=["label"])["label"].values
y_val = pd.read_parquet(DATA_DIR/"validation.parquet", columns=["label"])["label"].values
y_ts = pd.read_parquet(DATA_DIR/"test_sample.parquet", columns=["label"])["label"].values

# ===== TF-IDF in batches =====
print("\n[2/5] HashingVectorizer + TF-IDF in batches...", flush=True)

hv = HashingVectorizer(
    analyzer="char_wb", ngram_range=(2, 3), n_features=2**16,  # 65536
    alternate_sign=False, dtype=np.float32,
)

# Process train in batches, accumulate sparse matrix
BATCH = 50000
t0 = time.time()

# First pass: hash vectorize in batches, then fit tfidf
svd_cache = CACHE_DIR / "svd_train.npy"
if svd_cache.exists():
    print("  Loading cached SVD train...", flush=True)
    X_svd_train = np.load(svd_cache)
    # Need to reload pipeline components
    pipeline = joblib.load(CACHE_DIR / "tfidf_svd_pipeline.joblib")
    tfidf_transformer = pipeline["tfidf"]
    svd = pipeline["svd"]
else:
    train_code = pd.read_parquet(DATA_DIR/"train.parquet", columns=["code"])["code"].values

    # Hash in batches, store results
    hash_batches = []
    for i in range(0, len(train_code), BATCH):
        batch = [s[:MAX_CODE_LEN] if s else "" for s in train_code[i:i+BATCH]]
        hash_batches.append(hv.transform(batch))
        del batch; gc.collect()
        print(f"  Hash batch {i//BATCH+1}/{(len(train_code)-1)//BATCH+1}", flush=True)
    del train_code; gc.collect()

    X_hash_train = sp.vstack(hash_batches, format="csr")
    del hash_batches; gc.collect()
    print(f"  Hash matrix: {X_hash_train.shape}, nnz={X_hash_train.nnz}", flush=True)

    # TF-IDF transform
    tfidf_transformer = TfidfTransformer(sublinear_tf=True)
    X_tfidf_train = tfidf_transformer.fit_transform(X_hash_train)
    del X_hash_train; gc.collect()

    # SVD
    print("  SVD (80 components)...", flush=True)
    svd = TruncatedSVD(n_components=80, random_state=42)
    X_svd_train = svd.fit_transform(X_tfidf_train).astype(np.float32)
    del X_tfidf_train; gc.collect()
    print(f"  Explained var: {svd.explained_variance_ratio_.sum():.4f}", flush=True)

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

# Transform val/ts
val_code = pd.read_parquet(DATA_DIR/"validation.parquet", columns=["code"])["code"].values
X_svd_val = code_to_svd(val_code); del val_code; gc.collect()
ts_code = pd.read_parquet(DATA_DIR/"test_sample.parquet", columns=["code"])["code"].values
X_svd_ts = code_to_svd(ts_code); del ts_code; gc.collect()

# ===== Combine =====
print("\n[3/5] Combining...", flush=True)
X_train = np.hstack([X_svd_train, X_hc_train]); del X_svd_train, X_hc_train
X_val = np.hstack([X_svd_val, X_hc_val]); del X_svd_val, X_hc_val
X_ts = np.hstack([X_svd_ts, X_hc_ts]); del X_svd_ts, X_hc_ts
gc.collect()
feat_names = [f"svd_{i}" for i in range(80)] + hc_names
print(f"  Final: {X_train.shape}", flush=True)

# ===== Train =====
print("\n[4/5] Training LightGBM...", flush=True)
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
print("\n[5/5] Evaluation...", flush=True)
val_pred = model.predict(X_val)
print(f"\n--- Validation ---")
print(f"  F1: {f1_score(y_val, val_pred):.4f}")
print(classification_report(y_val, val_pred, target_names=["Human", "AI"], digits=4))

ts_pred = model.predict(X_ts)
print(f"--- Test Sample ---")
print(f"  F1: {f1_score(y_ts, ts_pred):.4f}")
print(classification_report(y_ts, ts_pred, target_names=["Human", "AI"], digits=4))

imp = model.feature_importances_
print("--- Top 15 Features ---")
for rank, i in enumerate(np.argsort(imp)[::-1][:15], 1):
    print(f"  {rank:>2}. {feat_names[i]:<30s} {imp[i]:>6d}")

del X_val, X_ts; gc.collect()

# Test in small chunks
print("\nPredicting test...", flush=True)
test = pd.read_parquet(DATA_DIR/"test.parquet")
test_ids = test["ID"].values

CHUNK = 25000
all_preds = []
for i in range(0, len(test), CHUNK):
    chunk = test.iloc[i:i+CHUNK]
    X_hc = np.asarray(extract_features(chunk), dtype=np.float32)
    X_hc[~np.isfinite(X_hc)] = 0.0
    X_svd = code_to_svd(chunk["code"].values)
    preds = model.predict(np.hstack([X_svd, X_hc]))
    all_preds.append(preds)
    del X_hc, X_svd; gc.collect()
    print(f"  Chunk {i//CHUNK+1}/{(len(test)-1)//CHUNK+1}", flush=True)

test_pred = np.concatenate(all_preds)
pd.DataFrame({"ID": test_ids, "label": test_pred.astype(int)}).to_csv(
    DATA_DIR / "submission_tfidf_lgbm_advanced.csv", index=False
)
joblib.dump({"model": model, "hv": hv, "tfidf_transformer": tfidf_transformer,
             "svd": svd}, MODEL_DIR / "lgbm_tfidf_advanced.joblib")
print("\nDone!")
