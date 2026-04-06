# SemEval 2026 Task 13 Subtask A — Competition Plan

## Goal
Binary classification: Human (0) vs AI-generated (1) code. 500K test rows.

## Key Challenge
**Domain shift**: Train = Python 91.5%, C++ 4.7%, Java 3.9%. Test has unseen languages (C#, JS, Go, C, PHP).
=> Language-agnostic features & models are critical.

---

## Phase 1: EDA & Baseline ✅
- [x] EDA notebook with visualizations
- [x] 37 hand-crafted features (`features.py`) — top: has_tabs (corr=0.80), indent_consistency (0.54)
- [x] Features + LightGBM baseline → Val F1=0.980, TS F1=0.387
- [x] TF-IDF char n-grams + LogReg baseline → Val F1=0.947, TS F1=0.381

## Phase 2: Feature Engineering ✅
- [x] Statistical: char_entropy, unique_char_ratio, repeated_line_ratio, token_diversity
- [x] Style: indent_consistency, has_tabs, mixed_indent, trailing_whitespace, bracket_ratio, naming conventions
- [x] Structure: max_nesting_depth, avg_tokens_per_line, line_len_std/skew, blank_line_cluster_count
- [x] Language-agnostic subset (29 features, excluding Python-specific ones)
- [x] All in `features.py`, cached to `cache/hc_*.npy`

## Phase 3: Text-based Models ✅
- [x] Hashing TF-IDF + SVD(80) + features + LightGBM → Val F1=0.988, TS F1=0.386
- [x] CodeBERT fine-tuned (3 epochs, bs=64 effective) → **Val F1=0.9956**, TS F1=0.329
- [ ] Experiment with DeBERTa or other text transformers
- [ ] Compare: frozen embeddings vs fine-tuned

## Phase 4: Handle Domain Shift ⚠️ CRITICAL
- [x] Validated on test_sample — ALL models fail on unseen languages (TS F1 ≈ 0.33-0.39)
- [x] Per-language analysis: human recall near 0% for C, Go, PHP, Java, C# (all predicted as AI)
- [x] Agnostic features (29) + SVD → Val F1=0.985, TS F1=0.388 (no improvement)
- [x] Threshold tuning — no significant improvement
- [ ] Pseudo-labeling: predict test set, use high-confidence predictions as extra train data
- [ ] Data augmentation: variable renaming, comment stripping, reformatting
- [ ] Test-time augmentation if applicable

## Phase 5: Ensemble & Optimization
- [ ] Blend: LightGBM features + TF-IDF model + Transformer model
- [ ] Stacking: train meta-learner on out-of-fold predictions
- [ ] Threshold tuning on validation set
- [ ] CV strategy: stratified by language + label
- [ ] Analyze per-language performance, fix weak spots

## Phase 6: Final Submissions
- [ ] Select best single model (safe pick)
- [ ] Select best ensemble (aggressive pick)
- [ ] Verify submission format matches sample_submission.csv
- [ ] Final sanity checks: label distribution, no NaN, correct ID mapping

---

## Results Summary

| Model | Val F1 | Test Sample F1 | Submission File |
|-------|--------|----------------|-----------------|
| LightGBM (37 features) | 0.9800 | 0.3872 | submission_lgbm_baseline.csv |
| TF-IDF + LogReg | 0.9469 | 0.3811 | submission_tfidf_baseline.csv |
| TF-IDF SVD + Features + LGBM | 0.9879 | 0.3864 | submission_tfidf_lgbm_advanced.csv |
| Agnostic + SVD + LGBM | 0.9853 | 0.3881 | submission_phase4_combined.csv |
| **CodeBERT fine-tuned** | **0.9956** | 0.3287 | submission_codebert.csv |

**Key Finding**: All models achieve >97% val F1 but ~33-39% on test_sample (unseen languages).
The domain shift is the dominant challenge. Models overfit to Python-specific patterns.

## Environment Notes
- **Memory limit**: 10GB cgroup (not the full 2TB system RAM!)
- **GPU**: NVIDIA H100 80GB MIG
- **CPUs**: 224 cores
- Must use batch processing and aggressive gc for TF-IDF on 500K rows

---

## Progress Log

| Date | Action | Val F1 | TS F1 | Notes |
|------|--------|--------|-------|-------|
| 2026-04-06 | EDA complete | — | — | Domain shift is main challenge |
| 2026-04-06 | Phase 1 baselines | 0.980 | 0.387 | LightGBM best baseline |
| 2026-04-06 | Phase 2 features | — | — | 37 features in features.py |
| 2026-04-06 | Phase 3 TF-IDF+LGBM | 0.988 | 0.386 | SVD 80 dims, marginal gain |
| 2026-04-06 | Phase 3 CodeBERT | 0.996 | 0.329 | Best val, worst domain shift |
| 2026-04-06 | Phase 4 agnostic | 0.985 | 0.388 | Removing Python features didn't help |
