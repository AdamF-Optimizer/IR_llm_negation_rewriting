import json
import os
import time
from typing import List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)

# QUEST gold file (json with query/docs/etc.)
GOLD_PATH = "C:/Users/C/Documents/Project_IR/DPR_package/data/QUEST_test_negations_final.json"

# QUEST corpus file (jsonl with {"title": ..., "text": ...})
CORPUS_PATH = "C:/Users/C/Documents/Project_IR/DPR_package/data/documents.jsonl"

# Output predictions
OUTPUT_PRED_PATH = "C:/Users/C/Documents/Project_IR/DPR_package/data/preds_dpr_quest_test.jsonl"

# FAISS index file for DPR context embeddings
INDEX_PATH = "index/quest_dpr_ctx.index"

# DPR models
QUESTION_MODEL_NAME = "facebook/dpr-question_encoder-multiset-base"
CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-multiset-base"

# QUESTION_MODEL_NAME = "facebook/dpr-question_encoder-single-nq-base"
# CONTEXT_MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"

# Retrieval parameters
BATCH_SIZE_CTX = 32   # batch size for encoding corpus
BATCH_SIZE_Q = 32     # batch size for encoding queries
MAX_LEN_CTX = 256     # max tokens for documents
MAX_LEN_Q = 64        # max tokens for queries
TOP_K_FINAL = 100     # what evaluation expects
INITIAL_MULT = 10     # 10 => retrieve 1000 when TOP_K_FINAL=100

APPLY_EXCLUSION = False  # set to False to completely disable exclusion logic

# configuration for exclusion behavior (only used if APPLY_EXCLUSION = True)
EXCLUSION_MODE = "penalize"  # "filter" / "penalize" / "boost"
EXCLUSION_WEIGHT = 0.5


def apply_exclusion(scores, documents, exclusion_criteria, mode, weight=0.5):
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


def load_dpr_models(
    q_model_name: str,
    ctx_model_name: str,
    device: torch.device,
):
    print("Loading DPR models...")
    q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(q_model_name)
    q_model = DPRQuestionEncoder.from_pretrained(q_model_name).to(device)
    q_model.eval()

    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(ctx_model_name)
    ctx_model = DPRContextEncoder.from_pretrained(ctx_model_name).to(device)
    ctx_model.eval()

    return q_tokenizer, q_model, ctx_tokenizer, ctx_model


def load_corpus_titles_and_texts(
    corpus_path: str,
) -> Tuple[List[str], List[str]]:
    """
    documents.jsonl: each line is {"title": ..., "text": ...}
    Returns:
        titles: list of titles
        texts: list of corresponding texts
    """
    titles = []
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            titles.append(obj["title"])
            texts.append(obj["text"])
    return titles, texts


def build_or_load_index(
    corpus_path: str,
    index_path: str,
    ctx_tokenizer,
    ctx_model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> Tuple[List[str], List[str], faiss.Index]:
    """
    Returns:
        titles: list of document titles, index i corresponds to FAISS vector i
        texts: list of corresponding document texts
        index:  FAISS inner-product index over DPR context embeddings
    """
    titles, texts = load_corpus_titles_and_texts(corpus_path)

    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from: {index_path}")
        index = faiss.read_index(index_path)
        return titles, texts, index

    print("Building FAISS index from scratch...")
    t0 = time.time()
    index = None

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
            batch_texts = texts[start: start + batch_size]
            inputs = ctx_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = ctx_model(**inputs)
            embs = outputs.pooler_output
            embs = F.normalize(embs, p=2, dim=1)
            embs_np = embs.cpu().numpy().astype("float32")

            if index is None:
                dim = embs_np.shape[1]
                index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)

            index.add(embs_np)

    index_dir = os.path.dirname(index_path)
    if index_dir and not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, index_path)
    print(f"Built index for {len(titles)} docs in {time.time() - t0:.1f}s")
    return titles, texts, index


def encode_queries(
    queries: List[str],
    q_tokenizer,
    q_model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 64,
) -> np.ndarray:
    all_embs = []

    with torch.no_grad():
        for start in tqdm(range(0, len(queries), batch_size), desc="Encoding queries"):
            batch_q = queries[start: start + batch_size]
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

    all_embs = torch.cat(all_embs, dim=0).numpy().astype("float32")
    return all_embs


def load_quest_examples(gold_path: str) -> list[dict]:
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return list(data)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load DPR models
    q_tokenizer, q_model, ctx_tokenizer, ctx_model = load_dpr_models(
        QUESTION_MODEL_NAME,
        CONTEXT_MODEL_NAME,
        device,
    )

    # 2) Build / load FAISS index over the QUEST corpus
    titles, texts, index = build_or_load_index(
        corpus_path=CORPUS_PATH,
        index_path=INDEX_PATH,
        ctx_tokenizer=ctx_tokenizer,
        ctx_model=ctx_model,
        device=device,
        batch_size=BATCH_SIZE_CTX,
        max_length=MAX_LEN_CTX,
    )

    # 3) Load gold QUEST examples (queries)
    print(f"Loading QUEST gold examples from {GOLD_PATH}")
    gold_examples = load_quest_examples(GOLD_PATH)
    queries = [ex["rewritten_query"] for ex in gold_examples]

    # 4) Encode queries with DPR question encoder
    q_embs = encode_queries(
        queries=queries,
        q_tokenizer=q_tokenizer,
        q_model=q_model,
        device=device,
        batch_size=BATCH_SIZE_Q,
        max_length=MAX_LEN_Q,
    )

    # 5) FAISS search
    print(f"Searching top-{TOP_K_FINAL} documents for each query...")
    initial_k = TOP_K_FINAL * INITIAL_MULT
    distances, indices = index.search(q_embs, initial_k)
    indices = indices.tolist()
    distances = distances.tolist()

    # 6) Write predictions in QUEST format
    print(f"Writing predictions to {OUTPUT_PRED_PATH}")
    out_dir = os.path.dirname(OUTPUT_PRED_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_PRED_PATH, "w", encoding="utf-8") as fout:
        for ex, doc_ids, sims in tqdm(
            zip(gold_examples, indices, distances),
            total=len(gold_examples),
            desc="Writing pred jsonl",
        ):
            exclusion_criteria = ex.get("exclusion_criteria", [])
            retrieved_texts = [texts[i] for i in doc_ids]

            sims_np = np.array(sims, dtype=np.float32)

            if APPLY_EXCLUSION:
                sims_np = apply_exclusion(
                    scores=sims_np,
                    documents=retrieved_texts,
                    exclusion_criteria=exclusion_criteria,
                    mode=EXCLUSION_MODE,
                    weight=EXCLUSION_WEIGHT,
                )

                # rerank only when exclusion changed scores
                order = np.argsort(sims_np)[::-1]
                doc_ids = [doc_ids[i] for i in order]
                sims_np = sims_np[order]

            # keep top-k final
            doc_ids = doc_ids[:TOP_K_FINAL]
            sims_np = sims_np[:TOP_K_FINAL]

            pred_titles = [titles[i] for i in doc_ids]
            pred_scores = sims_np.tolist()

            pred_ex = {
                "query": ex["query"],
                "docs": pred_titles,
                "original_query": ex.get("original_query", None),
                "scores": pred_scores,
                "metadata": ex.get("metadata", {}),
            }

            fout.write(json.dumps(pred_ex, ensure_ascii=False) + "\n")

    print("Done. You can now run the QUEST eval script.")

if __name__ == "__main__":
    main()