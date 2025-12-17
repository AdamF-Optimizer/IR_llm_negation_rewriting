import json
from collections import defaultdict


def load_examples(path):
    """Load examples from either JSON or JSONL format.

    Each example must contain:
      - query: str
      - docs: list of strings (titles)
      - metadata.template (optional)
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Case 1: JSON list
    if content.startswith("["):
        data = json.loads(content)
        return data

    # Case 2: JSONL
    examples = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(json.loads(line))
    return examples


def get_template(example):
    """Returns the template name if available, else 'unknown'."""
    meta = example.get("metadata", {})
    return meta.get("template", "unknown")


def compute_precision_recall_f1(gold_docs, pred_docs):
    gold = set(gold_docs)
    pred = set(pred_docs)

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def print_avg(examples, values, metric_name):
    avg = sum(values) / len(values)
    print(f"{metric_name}: {avg:.4f}  (over {len(values)} examples)")


def print_avg_by_template(examples, values, metric_name):
    buckets = defaultdict(list)
    for ex, val in zip(examples, values):
        buckets[get_template(ex)].append(val)

    print(f"{metric_name} by template:")
    for template, vals in buckets.items():
        avg = sum(vals) / len(vals)
        print(f"  {template}: {avg:.4f}  (n={len(vals)})")
    print()


def evaluate(gold_path, pred_path):
    print("Loading gold examples...")
    gold = load_examples(gold_path)
    print(f"Gold examples: {len(gold)}")

    print("Loading predicted examples...")
    pred = load_examples(pred_path)
    print(f"Predicted examples: {len(pred)}")

    # Map by query string (QUEST uses this as key)
    pred_map = {ex["query"]: ex for ex in pred}

    precision_list = []
    recall_list = []
    f1_list = []

    for gold_ex in gold:
        query = gold_ex["query"]
        if query not in pred_map:
            raise ValueError(f"Missing prediction for query: {query}")

        pred_ex = pred_map[query]

        p, r, f = compute_precision_recall_f1(
            gold_ex["docs"],
            pred_ex["docs"]
        )

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)

    print("\n=== RESULTS ===\n")
    print_avg(gold, precision_list, "Avg Precision")
    print_avg_by_template(gold, precision_list, "Avg Precision")

    print_avg(gold, recall_list, "Avg Recall")
    print_avg_by_template(gold, recall_list, "Avg Recall")

    print_avg(gold, f1_list, "Avg F1")
    print_avg_by_template(gold, f1_list, "Avg F1")


if __name__ == "__main__":
    # Edit these two lines before running:
    GOLD = "/QUEST_test_negations_final.json"
    PRED = "/preds_dpr_quest_test.jsonl"

    evaluate(GOLD, PRED)