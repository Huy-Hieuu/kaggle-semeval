"""Baseline 2: TF-IDF char n-grams + Logistic Regression (memory-optimized)"""
import gc, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")
DATA_DIR = "Task_A"
MAX_CODE_LEN = 2000

print("=" * 70)
print("  Baseline 2: TF-IDF Char N-grams + Logistic Regression")
print("=" * 70, flush=True)

# Load train only first
t0 = time.time()
train = pd.read_parquet(f"{DATA_DIR}/train.parquet", columns=["code", "label"])
y_train = train["label"].values
train_code = [s[:MAX_CODE_LEN] if s else "" for s in train["code"].values]
del train; gc.collect()
print(f"Loaded train in {time.time() - t0:.1f}s", flush=True)

# Fit TF-IDF on train
print("\nFitting TF-IDF (char_wb, 2-4 grams, 30K features)...", flush=True)
t0 = time.time()
tfidf = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 4),
    max_features=30000, sublinear_tf=True, dtype=np.float32,
)
X_train = tfidf.fit_transform(train_code)
del train_code; gc.collect()
print(f"  Train: {X_train.shape} ({time.time() - t0:.1f}s)", flush=True)

# Train LR
print("\nTraining Logistic Regression...", flush=True)
t0 = time.time()
model = LogisticRegression(C=1.0, max_iter=500, solver="saga", n_jobs=32, random_state=42)
model.fit(X_train, y_train)
del X_train, y_train; gc.collect()
print(f"  Training time: {time.time() - t0:.1f}s", flush=True)

# Evaluate on validation
print("\n--- Validation ---", flush=True)
val = pd.read_parquet(f"{DATA_DIR}/validation.parquet", columns=["code", "label"])
y_val = val["label"].values
val_code = [s[:MAX_CODE_LEN] if s else "" for s in val["code"].values]
del val; gc.collect()
X_val = tfidf.transform(val_code); del val_code; gc.collect()
val_pred = model.predict(X_val); del X_val; gc.collect()
print(f"  Accuracy: {accuracy_score(y_val, val_pred):.4f}")
print(f"  F1 Score: {f1_score(y_val, val_pred):.4f}")
print(classification_report(y_val, val_pred, target_names=["human", "AI"], digits=4))

# Evaluate on test_sample
print("--- Test Sample ---", flush=True)
ts = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet", columns=["code", "label"])
y_ts = ts["label"].values
ts_code = [s[:MAX_CODE_LEN] if s else "" for s in ts["code"].values]
del ts; gc.collect()
X_ts = tfidf.transform(ts_code); del ts_code; gc.collect()
ts_pred = model.predict(X_ts); del X_ts; gc.collect()
print(f"  Accuracy: {accuracy_score(y_ts, ts_pred):.4f}")
print(f"  F1 Score: {f1_score(y_ts, ts_pred):.4f}")
print(classification_report(y_ts, ts_pred, target_names=["human", "AI"], digits=4))

# Predict test in chunks
print("\nPredicting on test set (chunked)...", flush=True)
test = pd.read_parquet(f"{DATA_DIR}/test.parquet", columns=["ID", "code"])
test_ids = test["ID"].values
test_code = test["code"].values
del test; gc.collect()

CHUNK = 50000
all_preds = []
for i in range(0, len(test_code), CHUNK):
    chunk = [s[:MAX_CODE_LEN] if s else "" for s in test_code[i:i+CHUNK]]
    X_chunk = tfidf.transform(chunk)
    preds = model.predict(X_chunk)
    all_preds.append(preds)
    del X_chunk, chunk; gc.collect()
    print(f"  Chunk {i//CHUNK + 1}/{(len(test_code)-1)//CHUNK + 1} done", flush=True)

test_pred = np.concatenate(all_preds)
pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(
    f"{DATA_DIR}/submission_tfidf_baseline.csv", index=False
)
joblib.dump(model, "model_tfidf_baseline.joblib")
joblib.dump(tfidf, "vectorizer_tfidf_baseline.joblib")
print("\nSaved submission + model + vectorizer")
