import json
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
import torch
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

DATA_PATH = Path(".../NevIR_test_final.json")  # adjust if needed

QUESTION_MODEL_NAME = "facebook/dpr-question_encoder-multiset-base"
CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-multiset-base" 

# QUESTION_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base" 
# CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with DATA_PATH.open("r", encoding="utf-8") as f:
    raw_data = json.load(f)

examples = []

for ex in raw_data:
    #q1 = ex["q1"]
    #q2 = ex["q2"]
    q1 = ex["rewritten_query_q1"]
    q2 = ex["rewritten_query_q2"]
    d1 = ex["doc1"]
    d2 = ex["doc2"]

    # Example 1: q1, doc1 (relevant) vs doc2 (non-relevant)
    examples.append(
        {
            "query": q1,
            "docs": [d1, d2],  # relevant first
            "gold_index": 0,
            "meta": {"id": ex["id"], "pair": "q1"},
        }
    )

    # Example 2: q2, doc2 (relevant) vs doc1 (non-relevant)
    examples.append(
        {
            "query": q2,
            "docs": [d2, d1],  # relevant first
            "gold_index": 0,
            "meta": {"id": ex["id"], "pair": "q2"},
        }
    )

print(f"Loaded {len(raw_data)} rows -> {len(examples)} query-doc-pair examples.")


print("Loading DPR models...")

q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(QUESTION_MODEL_NAME)
q_encoder = DPRQuestionEncoder.from_pretrained(QUESTION_MODEL_NAME).to(DEVICE)
q_encoder.eval()

ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CONTEXT_MODEL_NAME)
ctx_encoder = DPRContextEncoder.from_pretrained(CONTEXT_MODEL_NAME).to(DEVICE)
ctx_encoder.eval()

def encode_question(question: str) -> torch.Tensor:
    """Encode a single question string into a DPR embedding of shape (1, dim)."""
    inputs = q_tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = q_encoder(**inputs)
        emb = outputs.pooler_output  # (1, hidden_dim)

    # L2-normalized embeddings with dot-product similarity
    emb = F.normalize(emb, p=2, dim=1)
    return emb  # (1, dim)


def encode_contexts(ctxs: list[str]) -> torch.Tensor:
    """Encode a list of context documents into DPR embeddings of shape (N, dim)."""
    inputs = ctx_tokenizer(
        ctxs,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = ctx_encoder(**inputs)
        emb = outputs.pooler_output  # (N, hidden_dim)

    emb = F.normalize(emb, p=2, dim=1)
    return emb  # (N, dim)

def apply_exclusion(scores, documents, exclusion_criteria, mode, weight=0.5):
    """
    Applies a boost, penalty, or filter-out to scores for documents
    containing one of the terms in the exclusion_criteria list.
    """
    # If exclusion criteria not provided, don't alter the scores
    if not exclusion_criteria:
        return scores

    # Check all documents for exclusion_criteria terms
    matches = np.array([
        any(criteria.lower() in doc.lower() for criteria in exclusion_criteria)
        for doc in documents
    ], dtype=bool)

    if mode == 'boost':
        scores[matches] *= (1 + weight)
    elif mode == 'penalize':
        scores[matches] *= (1 - weight)
    elif mode == 'filter':
        scores[matches] = 0.0

    return scores

rows = []

APPLY_EXCLUSION = True  # set to False to ignore exclusion criteria 

EXCLUSION_MODE = "filter"  # or "penalize" / "boost"
EXCLUSION_WEIGHT = 0.5     

model_name = QUESTION_MODEL_NAME 

print("\nEvaluating DPR (pairwise) " + ("with exclusion criteria..." if APPLY_EXCLUSION else "without applying exclusion criteria..."))

for ex in tqdm(raw_data, desc="Paired accuracy eval"):
    q1 = ex["rewritten_query_q1"]
    q2 = ex["rewritten_query_q2"]
    d1 = ex["doc1"]
    d2 = ex["doc2"]

    exclusion_q1 = ex.get("exclusion_criteria_q1", [])
    exclusion_q2 = ex.get("exclusion_criteria_q2", [])

    q1_emb = encode_question(q1)                 # (1, dim)
    ctx_q1 = encode_contexts([d1, d2])           # (2, dim)
    scores_q1 = torch.matmul(ctx_q1, q1_emb.T).squeeze(1).cpu().numpy()

    # apply per-example exclusion criteria for q1 (optional)
    if APPLY_EXCLUSION:
        scores_q1 = apply_exclusion(
            scores=scores_q1,
            documents=[d1, d2],
            exclusion_criteria=exclusion_q1,
            mode=EXCLUSION_MODE,
            weight=EXCLUSION_WEIGHT,
        )

    pred_q1 = scores_q1.argmax()          # gold index is 0 (d1 is relevant)
    correct_q1 = (pred_q1 == 0)

    q2_emb = encode_question(q2)
    ctx_q2 = encode_contexts([d2, d1])
    scores_q2 = torch.matmul(ctx_q2, q2_emb.T).squeeze(1).cpu().numpy()

    # apply per-example exclusion criteria for q2 (optional)
    if APPLY_EXCLUSION:
        scores_q2 = apply_exclusion(
            scores=scores_q2,
            documents=[d2, d1],
            exclusion_criteria=exclusion_q2,
            mode=EXCLUSION_MODE,
            weight=EXCLUSION_WEIGHT,
        )

    pred_q2 = scores_q2.argmax()          # gold index is 0 (d2 is relevant)
    correct_q2 = (pred_q2 == 0)

    # pairwise correctness: both directions must be right
    pair_correct = 1.0 if (correct_q1 and correct_q2) else 0.0

    rows.append({
        "id": ex["id"],
        "q1": scores_q1,    # post-exclusion scores
        "q2": scores_q2,    # post-exclusion scores
        "score": pair_correct,
    })

model_results = rows
model_df = pd.DataFrame(model_results)

# convert score vectors to probabilities
model_df["q1_probs"] = model_df.q1.apply(lambda x: F.softmax(torch.tensor(x), dim=0).numpy())
model_df["q2_probs"] = model_df.q2.apply(lambda x: F.softmax(torch.tensor(x), dim=0).numpy())

# pairwise_score is True iff pair_correct == 1.0
model_df["pairwise_score"] = model_df.score.apply(lambda x: x == 1.0)
overall_score = model_df.pairwise_score.mean()

print(f"For model {model_name} the average score is {overall_score}")