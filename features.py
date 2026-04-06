"""
Feature engineering module for human vs AI code classification.
SemEval 2026 Task 13 Subtask A.

Usage:
    from features import extract_features
    features_df = extract_features(df)  # df must have a 'code' column
"""

import re
import math
from collections import Counter

import numpy as np
import pandas as pd


def _char_entropy(s: str) -> float:
    """Shannon entropy of character distribution."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_nesting_depth(code: str) -> int:
    """Max depth of nested braces/brackets/parens."""
    depth = 0
    max_depth = 0
    openers = set("({[")
    closers = set(")}]")
    for ch in code:
        if ch in openers:
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch in closers:
            depth -= 1
            if depth < 0:
                depth = 0
    return max_depth


def _string_literal_chars(code: str) -> int:
    """Approximate count of characters inside string literals."""
    total = 0
    # Triple-quoted strings first, then single-quoted
    for pattern in (r'"""[\s\S]*?"""', r"'''[\s\S]*?'''", r'"[^"\n]*?"', r"'[^'\n]*?'"):
        for m in re.finditer(pattern, code):
            # subtract the quote delimiters
            matched = m.group()
            if matched.startswith('"""') or matched.startswith("'''"):
                total += len(matched) - 6
            else:
                total += len(matched) - 2
    return max(total, 0)


def _extract_row_features(code: str) -> dict:
    """Extract all features from a single code string. Returns a dict."""

    code_len = len(code)
    if code_len == 0:
        # Return zeros for empty code
        return {k: 0 for k in _FEATURE_NAMES}

    lines = code.split("\n")
    line_count = len(lines)
    line_lengths = [len(l) for l in lines]
    non_empty_lines = [l for l in lines if l.strip()]
    non_empty_count = len(non_empty_lines)
    empty_count = line_count - non_empty_count

    # --- Basic (Phase 1) ---
    avg_line_len = np.mean(line_lengths) if line_lengths else 0.0
    max_line_len = max(line_lengths) if line_lengths else 0

    whitespace_count = sum(1 for c in code if c.isspace())
    digit_count = sum(1 for c in code if c.isdigit())

    # --- Statistical (Phase 2) ---
    char_ent = _char_entropy(code)
    unique_chars = len(set(code))
    unique_char_ratio = unique_chars / code_len

    # Repeated lines (among non-empty lines)
    if non_empty_count > 0:
        stripped = [l.strip() for l in non_empty_lines]
        line_counter = Counter(stripped)
        repeated = sum(c - 1 for c in line_counter.values() if c > 1)
        repeated_line_ratio = repeated / non_empty_count
    else:
        repeated_line_ratio = 0.0

    # Token stats (whitespace split)
    tokens = code.split()
    token_count = len(tokens)
    if token_count > 0:
        unique_tokens = len(set(tokens))
        token_diversity = unique_tokens / token_count
        avg_token_len = np.mean([len(t) for t in tokens])
    else:
        token_diversity = 0.0
        avg_token_len = 0.0

    # --- Style/Pattern (Phase 2) ---
    # Indent consistency: std dev of leading whitespace length per line
    leading_ws = []
    has_tab = False
    has_space_indent = False
    trailing_ws_count = 0
    comment_count = 0

    for l in lines:
        if not l.strip():
            continue
        lws = len(l) - len(l.lstrip())
        leading_ws.append(lws)
        if "\t" in l[:lws + 1] if lws > 0 else False:
            has_tab = True
        if lws > 0 and " " in l[:lws]:
            has_space_indent = True
        # Trailing whitespace
        if l != l.rstrip():
            trailing_ws_count += 1
        # Comment lines
        stripped_l = l.strip()
        if stripped_l.startswith("#") or stripped_l.startswith("//"):
            comment_count += 1

    indent_consistency = float(np.std(leading_ws)) if len(leading_ws) > 1 else 0.0
    has_tabs = 1 if has_tab else 0
    mixed_indent = 1 if (has_tab and has_space_indent) else 0
    trailing_whitespace_ratio = trailing_ws_count / line_count if line_count > 0 else 0.0

    # Bracket / semicolon ratios
    bracket_count = sum(1 for c in code if c in "(){}[]")
    semicolon_count = code.count(";")
    comment_line_ratio = comment_count / line_count if line_count > 0 else 0.0

    has_docstring = 1 if ('"""' in code or "'''" in code) else 0

    # Naming conventions
    camelCase_count = len(re.findall(r"[a-z][a-zA-Z]*[A-Z][a-zA-Z]*", code))
    snake_case_count = len(re.findall(r"[a-z]+_[a-z]+", code))

    # --- Structure (Phase 2) ---
    max_nest = _max_nesting_depth(code)

    # Tokens per non-empty line
    if non_empty_count > 0:
        tokens_per_line = [len(l.split()) for l in non_empty_lines]
        avg_tokens_per_line = np.mean(tokens_per_line)
    else:
        avg_tokens_per_line = 0.0

    # Line length statistics
    ll_arr = np.array(line_lengths, dtype=np.float64)
    line_len_std = float(np.std(ll_arr)) if line_count > 1 else 0.0

    # Skewness (scipy-free)
    if line_count > 2 and line_len_std > 0:
        mean_ll = np.mean(ll_arr)
        line_len_skew = float(np.mean(((ll_arr - mean_ll) / line_len_std) ** 3))
    else:
        line_len_skew = 0.0

    # Blank line clusters
    blank_cluster_count = 0
    in_blank = False
    for l in lines:
        if not l.strip():
            if not in_blank:
                blank_cluster_count += 1
                in_blank = True
        else:
            in_blank = False

    # Code to comment ratio
    non_comment = non_empty_count - comment_count
    code_to_comment_ratio = non_comment / comment_count if comment_count > 0 else float(non_comment)

    # --- Character distribution (Phase 2) ---
    alpha_count = sum(1 for c in code if c.isalpha())
    upper_count = sum(1 for c in code if c.isupper())
    lower_count = sum(1 for c in code if c.islower())
    punct_count = sum(1 for c in code if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

    paren_count = code.count("(") + code.count(")")
    brace_count = code.count("{") + code.count("}")
    sq_bracket_count = code.count("[") + code.count("]")
    operator_count = sum(1 for c in code if c in "+-*/<>=!")

    str_lit_chars = _string_literal_chars(code)

    return {
        # Basic
        "code_len": code_len,
        "line_count": line_count,
        "avg_line_len": avg_line_len,
        "max_line_len": max_line_len,
        "empty_line_ratio": empty_count / line_count if line_count > 0 else 0.0,
        "whitespace_ratio": whitespace_count / code_len,
        "digit_ratio": digit_count / code_len,
        # Statistical
        "char_entropy": char_ent,
        "unique_char_ratio": unique_char_ratio,
        "repeated_line_ratio": repeated_line_ratio,
        "token_diversity": token_diversity,
        "avg_token_len": avg_token_len,
        # Style/Pattern
        "indent_consistency": indent_consistency,
        "has_tabs": has_tabs,
        "mixed_indent": mixed_indent,
        "trailing_whitespace_ratio": trailing_whitespace_ratio,
        "bracket_ratio": bracket_count / code_len,
        "semicolon_ratio": semicolon_count / line_count if line_count > 0 else 0.0,
        "comment_line_ratio": comment_line_ratio,
        "has_docstring": has_docstring,
        "camelCase_count": camelCase_count,
        "snake_case_count": snake_case_count,
        # Structure
        "max_nesting_depth": max_nest,
        "avg_tokens_per_line": avg_tokens_per_line,
        "line_len_std": line_len_std,
        "line_len_skew": line_len_skew,
        "blank_line_cluster_count": blank_cluster_count,
        "code_to_comment_ratio": code_to_comment_ratio,
        # Character distribution
        "alpha_ratio": alpha_count / code_len,
        "upper_ratio": upper_count / code_len,
        "lower_ratio": lower_count / code_len,
        "punct_ratio": punct_count / code_len,
        "paren_ratio": paren_count / code_len,
        "brace_ratio": brace_count / code_len,
        "square_bracket_ratio": sq_bracket_count / code_len,
        "operator_ratio": operator_count / code_len,
        "string_literal_ratio": str_lit_chars / code_len,
    }


# Canonical feature name list (used for empty-code fallback)
_FEATURE_NAMES = list(_extract_row_features("x = 1").keys())


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all features from a DataFrame with a 'code' column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'code' column with source code strings.

    Returns
    -------
    pd.DataFrame
        One column per feature, same index as input df.
    """
    # .apply returns a Series of dicts; pd.DataFrame converts to columns
    records = df["code"].fillna("").apply(_extract_row_features)
    features = pd.DataFrame(records.tolist(), index=df.index)
    return features


if __name__ == "__main__":
    import time

    DATA_DIR = "Task_A"

    print("Loading train.parquet ...")
    train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
    print(f"  Shape: {train.shape}")

    print("\nExtracting features ...")
    t0 = time.time()
    feats = extract_features(train)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(feats)} rows, {feats.shape[1]} features)")

    print("\n=== Feature Statistics ===")
    print(feats.describe().T.to_string())

    print("\n=== Top Feature Correlations with Label ===")
    corr = feats.corrwith(train["label"]).abs().sort_values(ascending=False)
    print(corr.head(20).to_string())

    out_path = f"{DATA_DIR}/train_features.parquet"
    print(f"\nSaving features to {out_path} ...")
    feats.to_parquet(out_path, index=False)
    print("Done.")
