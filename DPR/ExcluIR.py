import json
import numpy as np
import os
import time
from tqdm import tqdm
import faiss
import torch
import torch.nn.functional as F

from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)

from utils import *

#question_model_name_or_path = "facebook/dpr-question_encoder-multiset-base"
#context_model_name_or_path  = "facebook/dpr-ctx_encoder-multiset-base"

question_model_name_or_path = "facebook/dpr-question_encoder-single-nq-base"
context_model_name_or_path  = "facebook/dpr-ctx_encoder-single-nq-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

APPLY_EXCLUSION = False   # set to False to completely disable exclusion logic

# configuration for exclusion behavior (only used if APPLY_EXCLUSION = True)
EXCLUSION_MODE = "boost"   # "filter", "penalize", or "boost"
EXCLUSION_WEIGHT = 0.5


q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(question_model_name_or_path)
q_model = DPRQuestionEncoder.from_pretrained(question_model_name_or_path).to(device)
q_model.eval()

ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(context_model_name_or_path)
ctx_model = DPRContextEncoder.from_pretrained(context_model_name_or_path).to(device)
ctx_model.eval()

with open("C:/Users/C/Documents/Project_IR/DPR_package/data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open("C:/Users/C/Documents/Project_IR/DPR_package/data/ExcluIR_test_manual_final.json", "r", encoding="utf-8") as f:
    test_manual = json.load(f)

def encode_corpus(corpus_list, batch_size=32, max_length=256):
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_list), batch_size), desc="Encoding corpus"):
            batch_sentences = corpus_list[i : i + batch_size]
            inputs = ctx_tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = ctx_model(**inputs)
            embs = outputs.pooler_output
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).numpy().astype("float32")


def encode_queries(queries, batch_size=32, max_length=64):
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_q = queries[i : i + batch_size]
            inputs = q_tokenizer(
                batch_q,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = q_model(**inputs)
            embs = outputs.pooler_output
            embs = F.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0).numpy().astype("float32")


def apply_exclusion(scores, documents, exclusion_criteria, mode, weight=0.5):
    """
    Applies a boost, penalty, or filter-out to scores for documents
    containing one of the terms in the exclusion_criteria list.
    """
    if not exclusion_criteria:
        return scores

    matches = np.array([
        any(criteria.lower() in doc.lower() for criteria in exclusion_criteria)
        for doc in documents
    ], dtype=bool)

    if mode == "boost":
        scores[matches] *= (1 + weight)
    elif mode == "penalize":
        scores[matches] *= (1 - weight)
    elif mode == "filter":
        scores[matches] = 0.0

    return scores

collection = encode_corpus(corpus)

safe_index_name = context_model_name_or_path.replace("/", "_")
index_filename = f"index/{safe_index_name}.index"

if not os.path.exists(index_filename):
    os.makedirs(os.path.dirname(index_filename), exist_ok=True)
    dim = collection.shape[1]
    t = time.time()
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(collection)
    faiss.write_index(index, index_filename)
    print(f"Built index of {len(collection)} docs in {time.time() - t:.2f}s")
else:
    index = faiss.read_index(index_filename)
    print(f"Loaded existing index from {index_filename}")

queries = [sample["rewritten_query"] for sample in test_manual]
q_embs = encode_queries(queries)

k = len(corpus)

result_list = []
right_list = []
right_list_pos = []
right_list_neg = []

for num in tqdm(range(len(q_embs)), desc="Searching per query"):
    D, I = index.search(q_embs[num:num+1], k)
    doc_indices = I[0].tolist()
    scores = D[0].copy()

    docs = [corpus[idx] for idx in doc_indices]
    exclusion_criteria = test_manual[num].get("exclusion_criteria", [])

    if APPLY_EXCLUSION:
        scores = apply_exclusion(
            scores=scores,
            documents=docs,
            exclusion_criteria=exclusion_criteria,
            mode=EXCLUSION_MODE,
            weight=EXCLUSION_WEIGHT,
        )

    order = np.argsort(scores)[::-1]
    adjusted_indices = [doc_indices[i] for i in order]
    result_list.append(adjusted_indices)

    gt = test_manual[num]["corpus_sub_index"]
    right_list.append(gt)
    right_list_pos.append(gt[1:])
    right_list_neg.append(gt[:1])


recall_pos = compute_recall(result_list, right_list_pos)
recall_neg = compute_recall(result_list, right_list_neg)
mrr_pos = compute_MRR(result_list, right_list_pos)
mrr_neg = compute_MRR(result_list, right_list_neg)
rr = compute_right_rank(result_list, right_list)

metric = {
    "R@1": round(recall_pos[0] * 100, 2),
    "MRR@10": round(mrr_pos * 100, 2),
    "ΔR@1": round(recall_pos[0] * 100, 2) - round(recall_neg[0] * 100, 2),
    "ΔMRR@10": round(mrr_pos * 100, 2) - round(mrr_neg * 100, 2),
    "RR": round(rr * 100, 2),
}

print(metric)
