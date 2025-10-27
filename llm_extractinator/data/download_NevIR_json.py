from datasets import load_dataset

# Load NevIR dataset from Hugging Face
dataset = load_dataset("orionweller/NevIR")

# Save each split as JSON
for split in dataset.keys():  # e.g., 'train', 'test', 'validation'
    dataset[split].to_json(f"NevIR_{split}.jsonl", orient="records", lines=True)
    print(f"Saved {split} split to NevIR_{split}.jsonl")