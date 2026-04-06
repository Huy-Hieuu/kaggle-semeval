# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition

**SemEval 2026 Task 13 Subtask A** — Binary classification of code snippets as human-written (label 0) or AI-generated (label 1).

- Evaluation metric: check competition page (likely F1 or accuracy)
- Submission format: CSV with columns `ID, label` (see `Task_A/sample_submission.csv`, 500K rows)

## Data

All data lives in `Task_A/`:

| File | Rows | Columns |
|------|------|---------|
| `train.parquet` | 500K | code, generator, label, language |
| `validation.parquet` | 100K | code, generator, label, language |
| `test.parquet` | 500K | ID, code |
| `test_sample.parquet` | 1K | code, generator, label, language |
| `sample_submission.csv` | 1K | ID, label (template) |

Key characteristics:
- Labels roughly balanced: 47.7% human / 52.3% machine
- Languages in train: Python (91.5%), C++ (4.7%), Java (3.9%)
- Test set likely contains unseen languages (Go, PHP, C#, C, JS)
- 35 different LLM generators in training data
- Machine code averages ~1053 chars vs ~600 for human code
- No missing values in any split

## Environment

- Python environment with standard ML/data science packages available
- Data files are large — use chunked reading or sampling when prototyping
- Use `pyarrow` or `pandas` to read parquet files
