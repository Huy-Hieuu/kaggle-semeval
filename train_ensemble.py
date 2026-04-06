"""
Smart ensemble with language detection for SemEval 2026 Task 13 Subtask A.

Key insight: models trained on Python/C++/Java fail on unseen languages
(C#, JS, Go, C, PHP) by predicting everything as AI. This ensemble uses
language detection to apply different strategies per language group.

Memory-conscious: batch processing + aggressive gc.collect() for 10GB limit.
"""

import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Task_A"
CACHE_DIR = BASE_DIR / "cache"

SEEN_LANGUAGES = {"python", "c++", "java"}
UNSEEN_LANGUAGES = {"c#", "javascript", "go", "c", "php", "unknown"}

# ============================================================================
# Step 1: Language Detection
# ============================================================================

def detect_language(code: str) -> str:
    """
    Detect programming language from code using keyword heuristics.
    Returns lowercase language name or 'unknown'.
    """
    if not code or not isinstance(code, str):
        return "unknown"

    scores = {
        "python": 0,
        "java": 0,
        "c++": 0,
        "c#": 0,
        "javascript": 0,
        "go": 0,
        "c": 0,
        "php": 0,
    }

    # --- Python ---
    if "def " in code:
        scores["python"] += 3
    if "import " in code and "import java." not in code and "import (" not in code:
        scores["python"] += 1
    if "print(" in code:
        scores["python"] += 2
    if "elif " in code:
        scores["python"] += 5
    if "self." in code:
        scores["python"] += 3
    if "__init__" in code:
        scores["python"] += 5
    if "lambda " in code:
        scores["python"] += 1
    if '"""' in code or "'''" in code:
        scores["python"] += 3
    if "except " in code or "except:" in code:
        scores["python"] += 3
    if "True" in code or "False" in code or "None" in code:
        scores["python"] += 1
    # Python typically has no semicolons at end of lines
    lines = code.split("\n")
    semicolon_lines = sum(1 for l in lines if l.rstrip().endswith(";"))
    if semicolon_lines == 0 and len(lines) > 3:
        scores["python"] += 2
    if "for " in code and " in " in code and ":" in code:
        scores["python"] += 2

    # --- Java ---
    if "public class " in code:
        scores["java"] += 5
    if "System.out" in code:
        scores["java"] += 5
    if "public static void main" in code:
        scores["java"] += 8
    if "import java." in code:
        scores["java"] += 8
    if "private " in code or "protected " in code:
        scores["java"] += 1
    if "@Override" in code:
        scores["java"] += 4
    if "throws " in code:
        scores["java"] += 3
    if "new " in code and "{" in code:
        scores["java"] += 1
    if "String[] " in code or "String " in code:
        scores["java"] += 1
    if "ArrayList" in code or "HashMap" in code:
        scores["java"] += 3

    # --- C++ ---
    if "#include" in code:
        scores["c++"] += 3
        scores["c"] += 3
    if "cout" in code or "cin" in code:
        scores["c++"] += 5
    if "std::" in code:
        scores["c++"] += 6
    if "using namespace" in code:
        scores["c++"] += 6
    if "nullptr" in code:
        scores["c++"] += 5
    if "vector<" in code or "map<" in code or "set<" in code:
        scores["c++"] += 4
    if "template" in code and "<" in code:
        scores["c++"] += 4
    if "::" in code and "std" not in code:
        scores["c++"] += 2
    if "class " in code and "{" in code and "public:" in code:
        scores["c++"] += 4

    # --- C# ---
    if "using System" in code:
        scores["c#"] += 8
    if "namespace " in code and "{" in code:
        scores["c#"] += 3
    if "Console.Write" in code:
        scores["c#"] += 6
    if "Console.Read" in code:
        scores["c#"] += 6
    if "string " in code.lower() and "var " in code:
        scores["c#"] += 2
    if "get;" in code or "set;" in code:
        scores["c#"] += 5
    if "IEnumerable" in code or "IList" in code:
        scores["c#"] += 5
    if "async " in code and "await " in code and "Task" in code:
        scores["c#"] += 3
    if "LINQ" in code or ".Select(" in code or ".Where(" in code:
        scores["c#"] += 4
    if "public " in code and "namespace " in code:
        scores["c#"] += 2

    # --- JavaScript ---
    if "const " in code:
        scores["javascript"] += 2
    if "let " in code:
        scores["javascript"] += 2
    if "=>" in code:
        scores["javascript"] += 2
    if "console.log" in code:
        scores["javascript"] += 6
    if "require(" in code:
        scores["javascript"] += 5
    if "document." in code:
        scores["javascript"] += 6
    if "module.exports" in code:
        scores["javascript"] += 6
    if "function " in code and "(" in code and "{" in code:
        scores["javascript"] += 1
    if "undefined" in code or "null" in code:
        scores["javascript"] += 1
    if "addEventListener" in code:
        scores["javascript"] += 5
    if "Promise" in code or ".then(" in code:
        scores["javascript"] += 2
    if "var " in code and "function" in code:
        scores["javascript"] += 2

    # --- Go ---
    if "func " in code:
        scores["go"] += 3
    if "package main" in code or "package " in code:
        scores["go"] += 5
    if "fmt." in code:
        scores["go"] += 6
    if 'import "' in code or "import (" in code:
        scores["go"] += 5
    if ":=" in code:
        scores["go"] += 4
    if "func (" in code and ") " in code:
        # Go method receiver syntax
        scores["go"] += 4
    if "go func" in code or "goroutine" in code:
        scores["go"] += 5
    if "chan " in code:
        scores["go"] += 5
    if "defer " in code:
        scores["go"] += 4
    if "nil" in code and "func " in code:
        scores["go"] += 2

    # --- C (plain C, not C++) ---
    if "#include <stdio.h>" in code:
        scores["c"] += 6
    if "#include <stdlib.h>" in code:
        scores["c"] += 5
    if "printf(" in code:
        scores["c"] += 4
    if "malloc(" in code or "free(" in code:
        scores["c"] += 5
    if "int main(" in code:
        scores["c"] += 2
    if "typedef " in code and "struct" in code:
        scores["c"] += 4
    if "NULL" in code and "#include" in code:
        scores["c"] += 2
    # If it looks like C but has C++ features, discount C
    if "cout" in code or "cin" in code or "std::" in code or "class " in code:
        scores["c"] -= 5

    # --- PHP ---
    if "<?php" in code:
        scores["php"] += 10
    if code.count("$") >= 3:
        scores["php"] += 4
    if "echo " in code:
        scores["php"] += 3
    if "->" in code and "$" in code:
        scores["php"] += 4
    if "array(" in code:
        scores["php"] += 3
    if "function " in code and "$" in code:
        scores["php"] += 3
    if "<?=" in code:
        scores["php"] += 5

    # Disambiguation: C vs C++
    # If both score high, prefer C++ if it has C++-specific features
    if scores["c"] > 0 and scores["c++"] > 0:
        if scores["c++"] >= scores["c"]:
            scores["c"] = 0  # It's C++, not C
        else:
            scores["c++"] = max(0, scores["c++"] - 3)

    # Disambiguation: Java vs C#
    # Both use public class, braces, etc.
    if scores["java"] > 0 and scores["c#"] > 0:
        if "import java." in code or "System.out" in code:
            scores["c#"] = max(0, scores["c#"] - 5)
        elif "using System" in code or "namespace" in code:
            scores["java"] = max(0, scores["java"] - 5)

    # Disambiguation: JavaScript vs other
    if scores["javascript"] > 0 and scores["go"] > 0:
        if "package " in code and "func " in code:
            scores["javascript"] = max(0, scores["javascript"] - 5)

    best_lang = max(scores, key=scores.get)
    best_score = scores[best_lang]

    if best_score < 3:
        return "unknown"

    return best_lang


def detect_languages_batch(codes: pd.Series) -> pd.Series:
    """Apply language detection to a Series of code strings."""
    return codes.apply(detect_language)


# ============================================================================
# Step 2: Load models and predictions
# ============================================================================

def load_codebert_probs(split="test"):
    """Load cached CodeBERT probabilities (prob of AI/label=1)."""
    if split == "test":
        path = CACHE_DIR / "codebert_test_probs.npy"
    elif split == "val":
        path = CACHE_DIR / "codebert_val_probs.npy"
    else:
        raise ValueError(f"Unknown split: {split}")
    probs = np.load(path).astype(np.float32)
    print(f"  Loaded CodeBERT {split} probs: shape={probs.shape}, "
          f"min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
    return probs


def load_lgbm_model():
    """Load the LGBM baseline model."""
    model = joblib.load(BASE_DIR / "model_lgbm_baseline.joblib")
    print(f"  Loaded LGBM model: {type(model).__name__}, "
          f"n_features={model.n_features_in_}")
    return model


def predict_lgbm_chunked(model, df, chunk_size=25000):
    """
    Predict with LGBM in chunks to save memory.
    Returns array of probabilities (prob of AI/label=1).
    """
    from features import extract_features

    n = len(df)
    probs = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = df.iloc[start:end]
        feats = extract_features(chunk)
        chunk_probs = model.predict_proba(feats)[:, 1]
        probs[start:end] = chunk_probs.astype(np.float32)
        del feats, chunk_probs, chunk
        gc.collect()
        print(f"    LGBM chunk {start}-{end} done ({end}/{n})")

    return probs


def predict_lgbm_from_cache(model, cache_path):
    """
    Predict with LGBM using cached feature arrays.
    """
    feats = np.load(cache_path)
    print(f"  Loaded cached features: shape={feats.shape} from {cache_path}")
    probs = model.predict_proba(feats)[:, 1].astype(np.float32)
    del feats
    gc.collect()
    return probs


# ============================================================================
# Step 3: Ensemble strategies
# ============================================================================

def is_seen_language(lang):
    """Check if language was in training set."""
    return lang.lower() in SEEN_LANGUAGES


def ensemble_smart(codebert_probs, lgbm_probs, languages,
                   seen_threshold=0.5,
                   unseen_codebert_threshold=0.99,
                   require_lgbm_agree=True):
    """
    Smart ensemble strategy:
    - SEEN languages: use CodeBERT with standard threshold
    - UNSEEN languages: very conservative — only predict AI if CodeBERT is
      very confident AND LGBM agrees (if require_lgbm_agree=True)
    """
    n = len(codebert_probs)
    preds = np.zeros(n, dtype=np.int32)

    seen_mask = np.array([is_seen_language(l) for l in languages])
    unseen_mask = ~seen_mask

    # Seen languages: CodeBERT threshold=0.5
    preds[seen_mask] = (codebert_probs[seen_mask] > seen_threshold).astype(np.int32)

    # Unseen languages: very conservative
    if require_lgbm_agree:
        unseen_ai = ((codebert_probs > unseen_codebert_threshold) &
                     (lgbm_probs > 0.5))
        preds[unseen_mask] = unseen_ai[unseen_mask].astype(np.int32)
    else:
        preds[unseen_mask] = (codebert_probs[unseen_mask] > unseen_codebert_threshold).astype(np.int32)

    return preds


def strategy_codebert_only(codebert_probs, languages,
                           seen_threshold=0.5, unseen_threshold=0.99):
    """Strategy A: CodeBERT only, different thresholds by language group."""
    n = len(codebert_probs)
    preds = np.zeros(n, dtype=np.int32)
    seen_mask = np.array([is_seen_language(l) for l in languages])
    preds[seen_mask] = (codebert_probs[seen_mask] > seen_threshold).astype(np.int32)
    preds[~seen_mask] = (codebert_probs[~seen_mask] > unseen_threshold).astype(np.int32)
    return preds


def strategy_human_for_unseen(codebert_probs, languages, seen_threshold=0.5):
    """Strategy B: predict human for ALL unseen languages, CodeBERT for seen."""
    n = len(codebert_probs)
    preds = np.zeros(n, dtype=np.int32)  # default 0 = human
    seen_mask = np.array([is_seen_language(l) for l in languages])
    preds[seen_mask] = (codebert_probs[seen_mask] > seen_threshold).astype(np.int32)
    # Unseen: all 0 (human) — already initialized
    return preds


def strategy_agreement(codebert_probs, lgbm_probs, threshold=0.5):
    """Strategy C: both CodeBERT AND LGBM must agree on AI."""
    preds = ((codebert_probs > threshold) & (lgbm_probs > threshold)).astype(np.int32)
    return preds


# ============================================================================
# Step 4: Evaluation
# ============================================================================

def evaluate_predictions(y_true, y_pred, languages=None, label="Model"):
    """Print detailed evaluation metrics, overall and per-language."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report
    )

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)

    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    if languages is not None:
        print(f"\n  Per-language breakdown:")
        print(f"  {'Language':<15} {'N':>5} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} "
              f"{'Human':>6} {'AI':>4} {'PredH':>6} {'PredA':>6}")
        print(f"  {'-'*85}")

        unique_langs = sorted(set(languages))
        for lang in unique_langs:
            mask = np.array([l == lang for l in languages])
            yt = y_true[mask]
            yp = y_pred[mask]
            n_total = mask.sum()
            n_human = (yt == 0).sum()
            n_ai = (yt == 1).sum()
            pred_human = (yp == 0).sum()
            pred_ai = (yp == 1).sum()
            lang_acc = accuracy_score(yt, yp)
            lang_f1 = f1_score(yt, yp, average="binary", zero_division=0)
            lang_prec = precision_score(yt, yp, average="binary", zero_division=0)
            lang_rec = recall_score(yt, yp, average="binary", zero_division=0)
            seen_str = "*" if is_seen_language(lang) else " "
            print(f"  {seen_str}{lang:<14} {n_total:>5} {lang_acc:>7.4f} {lang_f1:>7.4f} "
                  f"{lang_prec:>7.4f} {lang_rec:>7.4f} {n_human:>6} {n_ai:>4} "
                  f"{pred_human:>6} {pred_ai:>6}")

        print(f"\n  * = seen language (in training set)")

    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print(" Smart Ensemble with Language Detection")
    print(" SemEval 2026 Task 13 Subtask A")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Validate language detection on test_sample
    # ------------------------------------------------------------------
    print("\n[Step 1] Language Detection Validation on test_sample")
    print("-" * 50)

    test_sample = pd.read_parquet(DATA_DIR / "test_sample.parquet")
    print(f"  Loaded test_sample: {test_sample.shape}")
    print(f"  True language distribution:\n{test_sample['language'].value_counts().to_string()}")

    detected_langs = detect_languages_batch(test_sample["code"])
    test_sample["detected_lang"] = detected_langs

    # Compare detected vs true
    correct = (test_sample["language"].str.lower() == test_sample["detected_lang"]).sum()
    total = len(test_sample)
    print(f"\n  Language detection accuracy: {correct}/{total} = {correct/total:.4f}")

    # Show confusion
    print(f"\n  Detection confusion (true -> detected):")
    for true_lang in sorted(test_sample["language"].unique()):
        mask = test_sample["language"] == true_lang
        detected = test_sample.loc[mask, "detected_lang"].value_counts()
        det_str = ", ".join(f"{k}:{v}" for k, v in detected.items())
        print(f"    {true_lang:>12} ({mask.sum():>4}) -> {det_str}")

    # ------------------------------------------------------------------
    # Step 2: Load models
    # ------------------------------------------------------------------
    print("\n[Step 2] Loading models and cached predictions")
    print("-" * 50)

    lgbm_model = load_lgbm_model()

    # ------------------------------------------------------------------
    # Step 3: Evaluate on test_sample
    # ------------------------------------------------------------------
    print("\n[Step 3] Evaluating strategies on test_sample")
    print("-" * 50)

    y_true = test_sample["label"].values
    languages = test_sample["detected_lang"].values

    # Get LGBM predictions for test_sample (use cached features)
    cache_path = CACHE_DIR / "hc_test_sample.npy"
    print("  Computing LGBM predictions for test_sample...")
    lgbm_probs_sample = predict_lgbm_from_cache(lgbm_model, cache_path)

    # We don't have CodeBERT probs for test_sample directly,
    # so compute LGBM-only baselines and use LGBM as proxy
    # Actually, let's check if test_sample aligns with first 1000 of test
    # test_sample is a separate labeled subset — we need to compute CodeBERT for it
    # Since we don't have cached CodeBERT for test_sample, we'll use LGBM only for
    # test_sample evaluation, and note that CodeBERT will be used in the final submission.

    print("\n  NOTE: CodeBERT probs not cached for test_sample.")
    print("  Using LGBM probs as stand-in for test_sample evaluation.")
    print("  Final submission will use actual CodeBERT probs for full test set.\n")

    # Use LGBM probs as stand-in for CodeBERT on test_sample
    codebert_probs_sample = lgbm_probs_sample  # proxy

    # --- Baseline: LGBM threshold=0.5 for all ---
    baseline_preds = (lgbm_probs_sample > 0.5).astype(np.int32)
    evaluate_predictions(y_true, baseline_preds, languages,
                         label="Baseline: LGBM threshold=0.5 (all languages)")

    # --- Strategy A: Different thresholds by language group ---
    strat_a_preds = strategy_codebert_only(
        lgbm_probs_sample, languages,
        seen_threshold=0.5, unseen_threshold=0.99
    )
    results_a = evaluate_predictions(y_true, strat_a_preds, languages,
                                     label="Strategy A: LGBM seen=0.5, unseen=0.99")

    # --- Strategy B: Human for all unseen ---
    strat_b_preds = strategy_human_for_unseen(
        lgbm_probs_sample, languages, seen_threshold=0.5
    )
    results_b = evaluate_predictions(y_true, strat_b_preds, languages,
                                     label="Strategy B: LGBM for seen, Human for unseen")

    # --- Strategy C: Agreement (both must say AI) ---
    # With only LGBM, this is same as baseline. Use a stricter threshold instead.
    strat_c_preds = (lgbm_probs_sample > 0.7).astype(np.int32)
    results_c = evaluate_predictions(y_true, strat_c_preds, languages,
                                     label="Strategy C: LGBM threshold=0.7 (all)")

    # --- Smart ensemble ---
    smart_preds = ensemble_smart(
        lgbm_probs_sample, lgbm_probs_sample, languages,
        seen_threshold=0.5,
        unseen_codebert_threshold=0.99,
        require_lgbm_agree=True
    )
    results_smart = evaluate_predictions(y_true, smart_preds, languages,
                                         label="Smart Ensemble (LGBM proxy)")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Strategy Comparison Summary")
    print("=" * 60)
    all_results = {
        "Baseline (LGBM t=0.5)": {"accuracy": (baseline_preds == y_true).mean(),
                                    "f1": __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_true, baseline_preds)},
        "A: Diff thresholds":     {"accuracy": results_a["accuracy"], "f1": results_a["f1"]},
        "B: Human for unseen":    {"accuracy": results_b["accuracy"], "f1": results_b["f1"]},
        "C: LGBM t=0.7":         {"accuracy": results_c["accuracy"], "f1": results_c["f1"]},
        "Smart Ensemble":         {"accuracy": results_smart["accuracy"], "f1": results_smart["f1"]},
    }
    print(f"  {'Strategy':<30} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*52}")
    for name, res in all_results.items():
        print(f"  {name:<30} {res['accuracy']:>10.4f} {res['f1']:>10.4f}")

    del test_sample, lgbm_probs_sample, codebert_probs_sample
    del baseline_preds, strat_a_preds, strat_b_preds, strat_c_preds, smart_preds
    gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Generate submission on full test set
    # ------------------------------------------------------------------
    print("\n[Step 4] Generating submission on full test set")
    print("-" * 50)

    # Load CodeBERT test probabilities
    print("  Loading CodeBERT test probabilities...")
    codebert_test_probs = load_codebert_probs("test")

    # Process test set in chunks for LGBM predictions and language detection
    CHUNK_SIZE = 25000
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    n_test = len(test_df)
    print(f"  Loaded test set: {n_test} rows")

    # Extract IDs
    test_ids = test_df["ID"].values.copy()

    # Detect languages for full test set (chunked)
    print("  Detecting languages...")
    all_languages = []
    for start in range(0, n_test, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_test)
        chunk_langs = detect_languages_batch(test_df["code"].iloc[start:end])
        all_languages.extend(chunk_langs.tolist())
        if (start // CHUNK_SIZE) % 4 == 0:
            print(f"    Language detection: {end}/{n_test}")
    all_languages = np.array(all_languages)

    # Print detected language distribution
    lang_counts = pd.Series(all_languages).value_counts()
    print(f"\n  Detected language distribution in test set:")
    for lang, count in lang_counts.items():
        seen_str = "(SEEN)" if is_seen_language(lang) else "(UNSEEN)"
        print(f"    {lang:<15} {count:>7} ({100*count/n_test:.1f}%) {seen_str}")

    # LGBM predictions in chunks
    print("\n  Computing LGBM predictions in chunks...")
    from features import extract_features
    lgbm_test_probs = np.zeros(n_test, dtype=np.float32)
    for start in range(0, n_test, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_test)
        chunk = test_df.iloc[start:end]
        feats = extract_features(chunk)
        chunk_probs = lgbm_model.predict_proba(feats)[:, 1]
        lgbm_test_probs[start:end] = chunk_probs.astype(np.float32)
        del feats, chunk_probs, chunk
        gc.collect()
        if (start // CHUNK_SIZE) % 4 == 0:
            print(f"    LGBM chunk: {end}/{n_test}")

    # Free test_df code column memory
    del test_df
    gc.collect()

    # Apply smart ensemble
    print("\n  Applying smart ensemble strategy...")
    final_preds = ensemble_smart(
        codebert_test_probs, lgbm_test_probs, all_languages,
        seen_threshold=0.5,
        unseen_codebert_threshold=0.99,
        require_lgbm_agree=True
    )

    # Also compute other strategies for comparison
    preds_a = strategy_codebert_only(codebert_test_probs, all_languages, 0.5, 0.99)
    preds_b = strategy_human_for_unseen(codebert_test_probs, all_languages, 0.5)
    preds_c = strategy_agreement(codebert_test_probs, lgbm_test_probs, 0.5)

    # Print prediction distribution per strategy
    print(f"\n  Test set prediction distributions:")
    print(f"  {'Strategy':<35} {'Human':>8} {'AI':>8} {'AI%':>7}")
    print(f"  {'-'*60}")
    for name, preds in [("Smart Ensemble", final_preds),
                         ("A: CodeBERT diff thresholds", preds_a),
                         ("B: Human for unseen", preds_b),
                         ("C: Agreement (CB+LGBM)", preds_c),
                         ("CodeBERT raw (t=0.5)", (codebert_test_probs > 0.5).astype(int))]:
        n_human = (preds == 0).sum()
        n_ai = (preds == 1).sum()
        print(f"  {name:<35} {n_human:>8} {n_ai:>8} {100*n_ai/len(preds):>6.1f}%")

    # Per-language stats for smart ensemble
    print(f"\n  Smart ensemble per-language prediction distribution:")
    print(f"  {'Language':<15} {'N':>7} {'Human':>7} {'AI':>7} {'AI%':>7}")
    print(f"  {'-'*45}")
    for lang in lang_counts.index:
        mask = all_languages == lang
        n_lang = mask.sum()
        n_ai = final_preds[mask].sum()
        n_human = n_lang - n_ai
        print(f"  {lang:<15} {n_lang:>7} {n_human:>7} {n_ai:>7} {100*n_ai/n_lang:>6.1f}%")

    # Save submission
    submission = pd.DataFrame({"ID": test_ids, "label": final_preds})
    submission_path = DATA_DIR / "submission_ensemble.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\n  Submission saved to {submission_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Label distribution: {submission['label'].value_counts().to_dict()}")

    # Also save strategy B (often safest) as alternative
    submission_b = pd.DataFrame({"ID": test_ids, "label": preds_b})
    submission_b_path = DATA_DIR / "submission_ensemble_humanUnseen.csv"
    submission_b.to_csv(submission_b_path, index=False)
    print(f"  Alternative submission (Strategy B) saved to {submission_b_path}")

    # Cleanup
    del codebert_test_probs, lgbm_test_probs, all_languages
    del final_preds, preds_a, preds_b, preds_c
    del submission, submission_b
    gc.collect()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f" Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
