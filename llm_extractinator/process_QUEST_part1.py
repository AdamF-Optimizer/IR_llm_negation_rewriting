import math
import sys
import json
import re
import os
import shutil

# Testing flag - set to True to only process first object
TESTING_MODE = True

# Save original directory
original_dir = os.getcwd()

os.chdir("QUEST")

QUEST_data = ["data/QUEST_train.jsonl", "data/QUEST_val.jsonl", "data/QUEST_test.jsonl"]

def clean_nan(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def remove_mark_tags(text):
    if not isinstance(text, str):
        return text
    # Remove <mark> and </mark> tags
    return re.sub(r'</?mark>', '', text)

def clean_object(obj):
    # Recursively clean object to replace NaN with None
    if isinstance(obj, dict):
        return {k: clean_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_object(i) for i in obj]
    else:
        return clean_nan(obj)

def filter_invalid_entries_from_jsonl(jsonl_file):
    """Read JSONL file directly and filter invalid entries"""
    filtered_data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                obj = json.loads(line)
                # Filter based on metadata
                if obj.get("metadata", {}).get("relevance_ratings") is not None:
                    cleaned_obj = clean_object(obj)
                    cleaned_obj["original_query_cleaned"] = remove_mark_tags(obj.get("original_query", ""))
                    filtered_data.append(cleaned_obj)

                    if TESTING_MODE and len(filtered_data) >= 1:
                        print(f"[TESTING MODE]: Only processing first valid entry")
                        break
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} in {jsonl_file} due to JSON error: {e}")
                continue
    
    return filtered_data

filtered_data_by_path = {}

# Process JSONL files directly
for file_path in QUEST_data:
    print(f"Processing {file_path}")
    
    # Create filtered JSON path
    filtered_path = f"{file_path.replace('.jsonl', '.json')}"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filtered_path), exist_ok=True)
    
    # Filter entries from JSONL
    filtered_data = filter_invalid_entries_from_jsonl(f"{file_path}")
    filtered_data_by_path[filtered_path] = filtered_data
    
    # Save filtered data as JSON
    with open(filtered_path, "w", encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered data saved to {filtered_path} ({len(filtered_data)} entries)")

# Task description
description = "You will be given a natural language query. Your task is to perform a comprehensive negation analysis and rewriting according to the following steps:\n\n1. Negation Detection:\nAnalyze the text in the \"query\" field and detect all types of negation based on the following taxonomy:\n- Sentential negation (e.g., \"not\", \"no\", \"never\")\n- Exclusionary negation (e.g., \"except\", \"besides\")\n- Affixal negation (prefixes like \"un-\", \"dis-\")\n- Implicit negation (verbs or words implying negation like \"deny\", \"refuse\")\n- Contrasting antonyms (pairs of antonyms implying negation)\n\nFor each negation detected, return:\n- The type name from the taxonomy\n- The character span (start and end indices) in the query text\n- A confidence score between 0 and 1 indicating detection certainty\n- A boolean field \"is_negated\" that is true if any negation is detected.\n\n2. Negation Rewriting:\nIf possible, rewrite the query so that no negation or exclusion terms remain, but the original meaning is preserved.\nExample: \"Bread not containing milk\" to \"Dairy-free bread\" or \"Vegan bread\".\n\nIf such rewriting is not possible, explicitly identify the exclusion criteria (terms or phrases to filter out after retrieval).\nExample: \"Cities in Germany except for Berlin\" rewritten_query: \"Cities in Germany\", exclusion_criteria: [\"Berlin\"].\n\n3. Output Format:\nRespond ONLY in JSON format with the following structure:\n{{\n  \"negation_analysis\": [\n    {{\n      \"type\": \"<negation type>\",\n      \"span\": [start_index, end_index],\n      \"confidence\": <float between 0 and 1>\n    }}\n  ],\n  \"is_negated\": <true/false>,\n  \"rewritten_query\": \"<rewritten positive query or original query>\",\n  \"exclusion_criteria\": [\"<list of terms or phrases to exclude>\"],\n  \"explanation\": \"<brief explanation of rewrite/filter decision>\"\n}}"

tasks_dir = "tasks"
if not os.path.exists(tasks_dir):
    os.makedirs(tasks_dir, exist_ok=True)
    print(f"Created directory: {os.path.abspath(tasks_dir)}")
else:
    print(f"Tasks directory already exists: {os.path.abspath(tasks_dir)}")


# Create task files
i = 0
for file_path, data in filtered_data_by_path.items():
    print(f"Creating task for {file_path} with {len(data)} entries.")
    task = {
        "Description": description,
        "Data_Path": file_path.replace("data/", ""),
        "Input_Field": "query",
        "Parser_Format": "negation_rewriting_full_parser.py"
    }
    task_file = f"tasks/Task{i:03d}.json"
    print(f"Creating task file: {task_file}")
    with open(task_file, "w", encoding='utf-8') as f:
        json.dump(task, f, indent=2)
    print(f"Created {task_file} at {file_path} with {len(data)} entries.")
    i += 1

# Copy parser file
parser_dest_dir = "tasks/parsers"
os.makedirs(parser_dest_dir, exist_ok=True)

# Source file is in the original directory
parser_source = os.path.join(original_dir, "Parsers", "negation_rewriting_full_parser.py")
parser_dest = os.path.join(parser_dest_dir, "negation_rewriting_full_parser.py")

print(f"\nCopying parser file...")
print(f"  Source: {parser_source}")
print(f"  Destination: {os.path.abspath(parser_dest)}")

try:
    if os.path.exists(parser_source):
        shutil.copy2(parser_source, parser_dest)
        print(f"Successfully copied parser file")
    else:
        print(f"ERROR: Source parser file not found: {parser_source}", file=sys.stderr)
except Exception as e:
    print(f"Failed to copy parser file: {e}", file=sys.stderr)

print("\n=== Summary ===")
if TESTING_MODE:
    print("TESTING MODE ENABLED - Only 1 entry per dataset processed")
print(f"Current directory: {os.getcwd()}")
print(f"Processed {len(filtered_data_by_path)} datasets")
print(f"Created {i} task files")
print("\nExtractinate commands to run from QUEST directory:")
for task_num in range(0, i):
    print(f"  extractinate --task_id {task_num:03d} --model_name qwen3:8b --num_predict 4096 --overwrite")
print("\nOutput will be found in: output/run/Task{XXX}-run0/nlp-predictions-dataset.json")

# Now run extractinate --task_id {task_id} --model_name qwen3:8b --num_predict 4096" for each task from within the QUEST folder:

# At the end, the output can be found in QUEST/output/run/Task{task_id}-run0/nlp-predictions-dataset.json
# Or, run process_QUEST_part2.py to collect and rename the outputs automatically.