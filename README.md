# AutoEIT Automated Scoring System
## GSoC 2026 Evaluation Test — HumanAI Foundation

[![Python](https://img.shields.io/badge/Python-3.10+-blue)]()
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## Overview

This project implements an automated scoring system for the **Elicited Imitation Task (EIT)** — a widely used research tool for measuring global language proficiency in second language learners.

The EIT requires learners to listen to and repeat sentences. Scoring is currently done manually by trained human raters — a slow, labor-intensive process. This system automates that scoring using a **3-layer hybrid pipeline** that is transparent, consistent, and reproducible.

---

## Architecture
```
Stimulus + Transcription
         ↓
┌────────┬────────┬────────┐
│  Rule  │  LLM   │  Sem.  │
│ Based  │ Scorer │  Sim.  │
│ (0-4)  │ (0-4)  │ (0-1)  │
└────────┴────────┴────────┘
         ↓
  Final Score (0-4)
  Rule×0.4 + LLM×0.4 + Sim×0.2
```


## 3-Layer Scoring Pipeline

### Layer 1 — Rule-Based Scorer
Transparent, explainable word-level matching:
- Normalizes text by removing accents and punctuation
- Filters disfluencies and filler words (`um`, `uh`, `...`)
- Matches words using **fuzzy matching** (`rapidfuzz`) to handle spelling variations
- Uses **Spanish stemming** (`nltk SnowballStemmer`) to handle verb form differences
- Maps word match percentage to 0-4 scale

### Layer 2 — Semantic Similarity
ML-supported meaning preservation check:
- Encodes stimulus and transcription using `paraphrase-multilingual-MiniLM-L12-v2`
- Computes **cosine similarity** in vector space
- Captures meaning preservation even when words differ completely
- Handles Spanish natively without cross-lingual bias

### Layer 3 — LLM Scorer
Nuanced edge case handler:
- Uses **Groq (Llama 3.1 8b)** with `temperature=0.0` for reproducible scoring
- Applies meaning-based rubric with Spanish linguistic understanding
- Handles disfluent speech, partial sentences, and fillers
- Provides human-readable reasoning for every score



## Scoring Rubric (0-4 scale)

| Score | Description |
|---|---|
| 0 | No meaningful words reproduced, meaning completely lost |
| 1 | Very few words correct, meaning mostly lost |
| 2 | Some words correct, partial meaning preserved |
| 3 | Most words correct, meaning mostly preserved |
| 4 | All or nearly all words correct, meaning fully preserved |



## Final Score Calculation
```
Final Score = (Rule Score × 0.4) + (LLM Score × 0.4) + (Similarity Score × 0.2)
```

- Rule-based and LLM get equal weight as primary scorers
- Semantic similarity acts as a supporting validation signal
- Score is clamped to 0-4 range
- **Divergence Flag** — raised when rule-based and LLM scores differ by more than 1 point, flagging the sentence for human review


## Project Structure
```
gsoc-2026-autoeit/
├── data/                    # input participant CSV files
├── output/                  # scored results
├── scorer.py                # 3-layer scoring pipeline
├── preprocessor.py          # data loading and cleaning
├── utils.py                 # helper functions
├── main.py                  # entry point
├── AutoEIT_Scoring.ipynb    # notebook with full output
├── AutoEIT_Scoring.pdf      # exported PDF of notebook
├── requirements.txt
└── README.md
```


## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/Ridzz110/gsoc-2026-autoeit
cd gsoc-2026-autoeit
```

**2. Create virtual environment**
```bash
python -m venv env
env\Scripts\activate  # Windows
source env/bin/activate  # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.env` file**
```
GROQ_API_KEY=your_groq_api_key
```

**5. Add participant CSV files to `data/` folder**

**6. Run**
```bash
python main.py
```


## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq — Llama 3.1 8b instant |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Fuzzy Matching | rapidfuzz |
| Stemming | nltk SnowballStemmer (Spanish) |
| Data Processing | pandas |
| Similarity | scikit-learn cosine similarity |


## Evaluation

### Validation approach:
- `temperature=0.0` ensures identical LLM scores on every run
- 3 independent layers cross-validate each other
- Divergence flagging identifies uncertain sentences for human review
- Fuzzy + stem matching handles spelling variations and verb forms

### Limitations:
- Without ground truth human scores, absolute accuracy cannot be measured
- Rule-based layer may underperform on heavy paraphrasing or synonyms
- Fuzzy matching threshold (80%) is heuristic


## Author

**Rida Batool**
AI Undergraduate | Mehran University of Engineering and Technology

GitHub: https://github.com/Ridzz110
LinkedIn: https://www.linkedin.com/in/ridabatool110
