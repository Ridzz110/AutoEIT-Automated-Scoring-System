import os
import re
import json
import time
import unicodedata
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from nltk.stem import SnowballStemmer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
stemmer = SnowballStemmer('spanish')
FILLERS = {'um', 'uh', 'eh', 'ah', '...'}


# ── Layer 1: Rule-Based ───────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'[¿?¡!.,]', '', text).lower().strip()

def words_match(w1: str, w2: str) -> bool:
    if w1 == w2: return True
    if fuzz.ratio(w1, w2) > 80: return True
    if stemmer.stem(w1) == stemmer.stem(w2): return True
    return False

def rule_based_score(stimulus: str, transcription: str) -> int:
    s_words = [w for w in normalize(stimulus).split() if w not in FILLERS]
    t_words = [w for w in normalize(transcription).split() if w not in FILLERS]
    matched = sum(1 for sw in s_words if any(words_match(sw, tw) for tw in t_words))
    pct = matched / len(s_words) if s_words else 0
    if pct >= 0.9: return 4
    elif pct >= 0.7: return 3
    elif pct >= 0.5: return 2
    elif pct >= 0.25: return 1
    else: return 0


# ── Layer 2: Semantic Similarity ──────────────────────────────────────────────
def semantic_similarity_score(stimulus: str, transcription: str) -> float:
    embeddings = embedding_model.encode([stimulus, transcription])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity), 4)


# ── Layer 3: LLM Scoring ──────────────────────────────────────────────────────
def llm_score(stimulus: str, transcription: str) -> dict:
    prompt = f"""You are an expert linguist scoring a Spanish Elicited Imitation Task (EIT).
Score how well the learner reproduced the meaning of the prompt sentence.

SCORING RUBRIC (0-4 scale):
0 — No meaningful words reproduced, meaning completely lost
1 — Very few words correct, meaning mostly lost
2 — Some words correct, partial meaning preserved
3 — Most words correct, meaning mostly preserved
4 — All or nearly all words correct, meaning fully preserved

Notes:
- Filler words like "um", "...", "uh" are ignored
- Minor spelling or pronunciation variations are acceptable
- Focus on MEANING preservation not exact wording
- IMPORTANT: reasoning MUST be in double quotes. Return valid JSON only.

PROMPT SENTENCE: {stimulus}
LEARNER UTTERANCE: {transcription}

Respond in this exact JSON format only:
{{
    "score": <number between 0 and 4>,
    "reasoning": "<one sentence explanation>"
}}"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"score": 0, "reasoning": "JSON parsing error"}
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print(f"  Rate limited, waiting 60s...")
                time.sleep(60)
            else:
                raise e
    return {"score": 0, "reasoning": "Max retries exceeded"}


# ── Final Score Combination ───────────────────────────────────────────────────
def calculate_final_score(rule: int, llm: int, sim: float) -> int:
    sim_score = round(sim * 4)
    final = round((rule * 0.4) + (llm * 0.4) + (sim_score * 0.2))
    return max(0, min(4, final))


# ── Main Scoring Function ─────────────────────────────────────────────────────
def score_utterance(stimulus: str, transcription: str) -> dict:
    rule = rule_based_score(stimulus, transcription)
    sim = semantic_similarity_score(stimulus, transcription)
    llm_result = llm_score(stimulus, transcription)
    llm = llm_result.get('score', 0)
    final = calculate_final_score(rule, llm, sim)

    return {
        "Rule_Score": rule,
        "LLM_Score": llm,
        "Semantic_Similarity": sim,
        "Score": final,
        "Divergence_Flag": abs(rule - llm) > 1,
        "Reasoning": llm_result.get('reasoning', '')
    }


# ── Participant Scoring ───────────────────────────────────────────────────────
def score_participant(df):
    results = {
        "Rule_Score": [], "LLM_Score": [], "Semantic_Similarity": [],
        "Score": [], "Divergence_Flag": [], "Reasoning": []
    }

    for _, row in df.iterrows():
        print(f"  Scoring: {row['Stimulus'][:50]}...")
        result = score_utterance(
            row['Stimulus'],
            str(row['Transcription Rater 1'])
        )
        for key in results:
            results[key].append(result[key])

    for key, values in results.items():
        df[key] = values

    return df