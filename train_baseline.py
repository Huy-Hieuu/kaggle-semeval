"""
Phase 1 baselines for SemEval 2026 Task 13 Subtask A:
Human vs AI code classification.

Baseline 1: Hand-crafted features + LightGBM
Baseline 2: TF-IDF char n-grams + LogisticRegression
"""

import gc
import time
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from features import extract_features

warnings.filterwarnings("ignore")

DATA_DIR = "Task_A"


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n", flush=True)


def evaluate(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"--- {dataset_name} ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=["human", "AI"], digits=4))
    return acc, f1


def run_baseline1():
    """Baseline 1: Hand-crafted features + LightGBM"""
    print_header("Baseline 1: Hand-crafted Features + LightGBM")

    # Load data
    t0 = time.time()
    train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    val = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
    test_sample = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")
    test = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    print(f"  Loaded data in {time.time() - t0:.1f}s", flush=True)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test_sample = test_sample["label"].values
    test_ids = test["ID"].values

    # Extract features
    print("Extracting features...", flush=True)
    t0 = time.time()
    X_train = extract_features(train)
    print(f"  Train: {X_train.shape} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    X_val = extract_features(val)
    print(f"  Val: {X_val.shape} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    X_ts = extract_features(test_sample)
    print(f"  Test sample: {X_ts.shape} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    X_test = extract_features(test)
    print(f"  Test: {X_test.shape} ({time.time() - t0:.1f}s)", flush=True)

    # Free raw data
    del train, val, test_sample, test
    gc.collect()

    # Train
    print("\nTraining LightGBM...", flush=True)
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)],
    )
    print(f"  Best iteration: {model.best_iteration_}")
    print(f"  Training time: {time.time() - t0:.1f}s", flush=True)

    # Evaluate
    val_pred = model.predict(X_val)
    val_acc, val_f1 = evaluate(y_val, val_pred, "Validation")

    ts_pred = model.predict(X_ts)
    ts_acc, ts_f1 = evaluate(y_test_sample, ts_pred, "Test Sample (domain shift)")

    # Feature importance
    print("--- Top 20 Feature Importances ---")
    feat_names = X_train.columns.tolist()
    imp = model.feature_importances_
    for rank, i in enumerate(np.argsort(imp)[::-1][:20], 1):
        print(f"  {rank:>2}. {feat_names[i]:<35s} {imp[i]:>6d}")

    # Save
    test_pred = model.predict(X_test)
    pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(
        f"{DATA_DIR}/submission_lgbm_baseline.csv", index=False
    )
    joblib.dump(model, "model_lgbm_baseline.joblib")
    print(f"\nSaved submission + model", flush=True)

    # Cleanup
    del X_train, X_val, X_ts, X_test, model
    gc.collect()

    return val_f1, ts_f1


def run_baseline2():
    """Baseline 2: TF-IDF char n-grams + Logistic Regression"""
    print_header("Baseline 2: TF-IDF Char N-grams + Logistic Regression")

    # Load data
    t0 = time.time()
    train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    val = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
    test_sample = pd.read_parquet(f"{DATA_DIR}/test_sample.parquet")
    test = pd.read_parquet(f"{DATA_DIR}/test.parquet")
    print(f"  Loaded data in {time.time() - t0:.1f}s", flush=True)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test_sample = test_sample["label"].values
    test_ids = test["ID"].values

    # Extract code strings, free DataFrames
    train_code = train["code"].fillna("").values
    val_code = val["code"].fillna("").values
    ts_code = test_sample["code"].fillna("").values
    test_code = test["code"].fillna("").values
    del train, val, test_sample, test
    gc.collect()

    # TF-IDF
    print("Fitting TF-IDF (char_wb, 2-5 grams, 100K features)...", flush=True)
    t0 = time.time()
    tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        max_features=100000, sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(train_code)
    print(f"  Train: {X_train.shape} ({time.time() - t0:.1f}s)", flush=True)
    del train_code; gc.collect()

    X_val = tfidf.transform(val_code)
    X_ts = tfidf.transform(ts_code)
    X_test = tfidf.transform(test_code)
    del val_code, ts_code, test_code; gc.collect()
    print(f"  All transforms done ({time.time() - t0:.1f}s)", flush=True)

    # Train
    print("\nTraining Logistic Regression...", flush=True)
    t0 = time.time()
    model = LogisticRegression(C=1.0, max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time() - t0:.1f}s", flush=True)

    # Evaluate
    val_pred = model.predict(X_val)
    val_acc, val_f1 = evaluate(y_val, val_pred, "Validation")

    ts_pred = model.predict(X_ts)
    ts_acc, ts_f1 = evaluate(y_test_sample, ts_pred, "Test Sample (domain shift)")

    # Save
    test_pred = model.predict(X_test)
    pd.DataFrame({"ID": test_ids, "label": test_pred}).to_csv(
        f"{DATA_DIR}/submission_tfidf_baseline.csv", index=False
    )
    joblib.dump(model, "model_tfidf_baseline.joblib")
    joblib.dump(tfidf, "vectorizer_tfidf_baseline.joblib")
    print(f"\nSaved submission + model + vectorizer", flush=True)

    del X_train, X_val, X_ts, X_test, model, tfidf
    gc.collect()

    return val_f1, ts_f1


def main():
    total_start = time.time()

    b1_val_f1, b1_ts_f1 = run_baseline1()
    b2_val_f1, b2_ts_f1 = run_baseline2()

    print_header("Baseline Comparison")
    print(f"{'Model':<35s} {'Val F1':>8s} {'TestSample F1':>14s}")
    print("-" * 60)
    print(f"{'LightGBM (features)':<35s} {b1_val_f1:>8.4f} {b1_ts_f1:>14.4f}")
    print(f"{'TF-IDF + LogReg':<35s} {b2_val_f1:>8.4f} {b2_ts_f1:>14.4f}")
    print(f"\nTotal runtime: {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
