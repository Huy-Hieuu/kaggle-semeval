#!/usr/bin/env python3
"""
OOD-Robust Model: Uses only cross-language, cross-domain invariant features.
Key insight from PDF: test has 4 scenarios including unseen domains (research/production).
AI code patterns that transfer across languages AND domains:
- More comments (AI over-explains)
- More blank lines (AI formats "perfectly")
- Fewer tabs (AI uses spaces consistently)
- More tokens per line (AI writes denser expressions sometimes)
- Lower camelCase count relative to code size
"""
import gc, time, warnings
import numpy as np
import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb

warnings.filterwarnings("ignore")
DATA_DIR = Path("Task_A")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("OOD-Robust Model: Cross-language/domain invariant features")
print("=" * 70, flush=True)

def extract_ood_features(df):
    """Extract features proven to work across ALL languages and domains."""
    codes = df['code'].fillna('').values
    features = []

    for code in codes:
        lines = code.split('\n')
        non_empty = [l for l in lines if l.strip()]
        n_lines = max(len(lines), 1)
        n_non_empty = max(len(non_empty), 1)
        code_len = max(len(code), 1)

        # Comment detection (works across languages)
        comment_lines = sum(1 for l in lines if l.strip().startswith(('#', '//', '*', '/*')))

        # Blank line patterns (AI adds more structured spacing)
        blank_lines = sum(1 for l in lines if not l.strip())
        blank_ratio = blank_lines / n_lines

        # Blank line clusters (consecutive blank lines)
        blank_clusters = 0
        in_blank = False
        for l in lines:
            if not l.strip():
                if not in_blank:
                    blank_clusters += 1
                    in_blank = True
            else:
                in_blank = False

        # Tab usage (human code more likely to have tabs)
        has_tabs = 1 if '\t' in code else 0
        tab_lines = sum(1 for l in lines if l.startswith('\t'))
        tab_ratio = tab_lines / n_lines

        # Trailing whitespace (human code has more trailing spaces)
        trailing_ws = sum(1 for l in non_empty if l != l.rstrip())
        trailing_ws_ratio = trailing_ws / n_non_empty

        # Comment ratio
        comment_ratio = comment_lines / n_lines

        # Docstring / block comments (AI tends to add more)
        has_docstring = 1 if ('"""' in code or "'''" in code or '/**' in code) else 0

        # Avg tokens per line
        tokens = [len(l.split()) for l in non_empty]
        avg_tokens_per_line = np.mean(tokens) if tokens else 0

        # Line length consistency (AI is more consistent)
        line_lens = [len(l) for l in non_empty]
        avg_line_len = np.mean(line_lens) if line_lens else 0
        std_line_len = np.std(line_lens) if len(line_lens) > 1 else 0
        cv_line_len = std_line_len / avg_line_len if avg_line_len > 0 else 0

        # Indentation consistency
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        indent_std = np.std(indents) if len(indents) > 1 else 0

        # Upper case ratio (human code has more CONSTANTS)
        upper_chars = sum(1 for c in code if c.isupper())
        upper_ratio = upper_chars / code_len

        # Punctuation density
        punct_chars = sum(1 for c in code if c in '(){}[]<>.,;:!@#$%^&*+-=|\\/?~`')
        punct_ratio = punct_chars / code_len

        # Operator ratio
        op_chars = sum(1 for c in code if c in '+-*/<>=!&|^~')
        op_ratio = op_chars / code_len

        # Bracket patterns
        paren = (code.count('(') + code.count(')')) / code_len
        brace = (code.count('{') + code.count('}')) / code_len

        # Code length features
        log_code_len = np.log1p(len(code))

        # Repetition ratio (duplicate lines)
        if non_empty:
            unique_lines = len(set(l.strip() for l in non_empty))
            repeat_ratio = 1 - unique_lines / n_non_empty
        else:
            repeat_ratio = 0

        # Character entropy
        from collections import Counter
        char_counts = Counter(code)
        total = sum(char_counts.values())
        if total > 0:
            probs = np.array(list(char_counts.values())) / total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
        else:
            entropy = 0

        # Token diversity
        all_tokens = code.split()
        n_tokens = max(len(all_tokens), 1)
        unique_tokens = len(set(all_tokens))
        token_diversity = unique_tokens / n_tokens

        # AI-specific: tends to have more function definitions per code length
        func_defs = len(re.findall(r'\b(def |func |function |fn )', code))
        func_density = func_defs / (code_len / 1000)  # per 1K chars

        # Whitespace ratio
        ws = sum(1 for c in code if c in ' \t\n\r')
        ws_ratio = ws / code_len

        # Digit ratio
        digits = sum(1 for c in code if c.isdigit())
        digit_ratio = digits / code_len

        # String literal ratio (approximate)
        in_str = False
        str_chars = 0
        quote_char = None
        for i, c in enumerate(code[:2000]):  # limit for speed
            if not in_str and c in '"\'':
                in_str = True; quote_char = c
            elif in_str and c == quote_char:
                in_str = False
            elif in_str:
                str_chars += 1
        str_ratio = str_chars / min(code_len, 2000)

        features.append({
            'comment_ratio': comment_ratio,
            'has_docstring': has_docstring,
            'blank_ratio': blank_ratio,
            'blank_clusters': blank_clusters,
            'has_tabs': has_tabs,
            'tab_ratio': tab_ratio,
            'trailing_ws_ratio': trailing_ws_ratio,
            'avg_tokens_per_line': avg_tokens_per_line,
            'avg_line_len': avg_line_len,
            'std_line_len': std_line_len,
            'cv_line_len': cv_line_len,
            'indent_std': indent_std,
            'upper_ratio': upper_ratio,
            'punct_ratio': punct_ratio,
            'op_ratio': op_ratio,
            'paren_ratio': paren,
            'brace_ratio': brace,
            'log_code_len': log_code_len,
            'repeat_ratio': repeat_ratio,
            'char_entropy': entropy,
            'token_diversity': token_diversity,
            'func_density': func_density,
            'ws_ratio': ws_ratio,
            'digit_ratio': digit_ratio,
            'str_ratio': str_ratio,
            'n_lines': n_lines,
        })

    result = pd.DataFrame(features)
    result = result.fillna(0)
    result = result.replace([np.inf, -np.inf], 0)
    return result

# ===== Train =====
print("\n[1/4] Extracting features...", flush=True)
t0 = time.time()
train = pd.read_parquet(DATA_DIR/"train.parquet")
X_train = extract_ood_features(train)
y_train = train['label'].values
del train; gc.collect()
print(f"  Train: {X_train.shape} ({time.time()-t0:.1f}s)", flush=True)

val = pd.read_parquet(DATA_DIR/"validation.parquet")
X_val = extract_ood_features(val)
y_val = val['label'].values
del val; gc.collect()

ts = pd.read_parquet(DATA_DIR/"test_sample.parquet")
X_ts = extract_ood_features(ts)
y_ts = ts['label'].values

print(f"\n[2/4] Training LightGBM...", flush=True)
t0 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=1500, learning_rate=0.05, num_leaves=31,  # smaller leaves = less overfit
    subsample=0.7, colsample_bytree=0.7, min_child_samples=100,  # more regularization
    reg_alpha=1.0, reg_lambda=1.0,  # strong regularization
    random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(
    X_train.values, y_train,
    eval_set=[(X_val.values, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)
del X_train; gc.collect()
print(f"  Best iter: {model.best_iteration_} ({time.time()-t0:.1f}s)", flush=True)

# ===== Evaluate =====
print(f"\n[3/4] Evaluation...", flush=True)
val_pred = model.predict(X_val.values)
val_prob = model.predict_proba(X_val.values)[:, 1]
print(f"\n--- Validation ---")
print(f"  F1: {f1_score(y_val, val_pred):.4f}")
print(classification_report(y_val, val_pred, target_names=["Human", "AI"], digits=4))

ts_pred = model.predict(X_ts.values)
ts_prob = model.predict_proba(X_ts.values)[:, 1]
print(f"--- Test Sample ---")
print(f"  F1: {f1_score(y_ts, ts_pred):.4f}")
print(classification_report(y_ts, ts_pred, target_names=["Human", "AI"], digits=4))

# Per-language
print("Per-language:")
for lang in sorted(ts['language'].unique()):
    mask = ts['language'].values == lang
    y_t = y_ts[mask]; y_p = ts_pred[mask]
    f1 = f1_score(y_t, y_p, zero_division=0)
    acc = accuracy_score(y_t, y_p)
    print(f"  {lang:<12s} Acc={acc:.3f} F1={f1:.3f}")

# Threshold sweep on test_sample
print(f"\nThreshold sweep on test_sample (macro F1):")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred_t = (ts_prob >= t).astype(int)
    mf1 = f1_score(y_ts, pred_t, average='macro')
    f1_ai = f1_score(y_ts, pred_t)
    print(f"  t={t:.1f}: macro_F1={mf1:.4f} F1_AI={f1_ai:.4f} %AI={pred_t.mean()*100:.1f}%")

# Feature importance
print("\n--- Feature Importances ---")
imp = model.feature_importances_
feat_names = X_val.columns.tolist()
for rank, i in enumerate(np.argsort(imp)[::-1][:15], 1):
    print(f"  {rank:>2}. {feat_names[i]:<25s} {imp[i]:>6d}")

del X_val, X_ts; gc.collect()

# ===== Test predictions =====
print(f"\n[4/4] Test predictions...", flush=True)
test = pd.read_parquet(DATA_DIR/"test.parquet")
CHUNK = 25000
all_probs = []
for i in range(0, len(test), CHUNK):
    chunk = test.iloc[i:i+CHUNK]
    X_chunk = extract_ood_features(chunk)
    probs = model.predict_proba(X_chunk.values)[:, 1]
    all_probs.append(probs)
    del X_chunk; gc.collect()
    if (i // CHUNK) % 5 == 0:
        print(f"  Chunk {i//CHUNK+1}/{(len(test)-1)//CHUNK+1}", flush=True)

test_probs = np.concatenate(all_probs)
np.save('cache/ood_robust_test_probs.npy', test_probs)

# Save with different thresholds
for t in [0.5, 0.6, 0.7]:
    pred = (test_probs >= t).astype(int)
    pd.DataFrame({'ID': test['ID'].values, 'label': pred}).to_csv(
        DATA_DIR / f'sub_ood_robust_t{t}.csv', index=False)
    print(f"  t={t}: {pred.sum()} AI ({pred.mean()*100:.1f}%)")

joblib.dump(model, MODEL_DIR / "ood_robust.joblib")
print("\nDone!")
