import json
import shutil
import subprocess
import sys
from sys import platform
import os

# Testing flag - set to True to only process first object
TESTING_MODE = True

# Save original directory
original_dir = os.getcwd()

os.chdir("NevIR")

# Split NevIR data into two, with q1 and q2 separately
nevir_files = ['data/NevIR_train.json', 'data/NevIR_test.json', 'data/NevIR_validation.json']

new_nevir_files = []


# Split each NevIR file into two files: one for q1/doc1 and one for q2/doc2
for nevir_file in nevir_files:
    # Load input file
    with open(nevir_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Limit to first object if testing
    if TESTING_MODE:
        data = data[:1]

    q1_objects = []
    q2_objects = []

    for obj in data:
        # Entry for q1/doc1
        q1_obj = {
            "id": obj["id"],
            "WorkerId": obj["WorkerId"],
            "q1": obj["q1"],
            "doc1": obj["doc1"]
        }
        q1_objects.append(q1_obj)
        
        # Entry for q2/doc2
        q2_obj = {
            "id": obj["id"],
            "WorkerId": obj["WorkerId"],
            "q2": obj["q2"],
            "doc2": obj["doc2"]
        }
        q2_objects.append(q2_obj)

    # Save the two lists as separate JSON files
    base_name = nevir_file.replace('.json', '')
    with open(f'{base_name}_q1.json', 'w', encoding='utf-8') as f:
        json.dump(q1_objects, f, ensure_ascii=False, indent=2)

    with open(f'{base_name}_q2.json', 'w', encoding='utf-8') as f:
        json.dump(q2_objects, f, ensure_ascii=False, indent=2)
    print(f"Split completed for {nevir_file}: '{base_name}_q1.json' and '{base_name}_q2.json' created.")
    new_nevir_files.append(f'{base_name}_q1.json')
    new_nevir_files.append(f'{base_name}_q2.json')

# Create the following task:
# {
#     "Description": "You will be given a natural language query. Your task is to perform a comprehensive negation analysis and rewriting according to the following steps:\n\n1. Negation Detection:\nAnalyze the text in the \"query\" field and detect all types of negation based on the following taxonomy:\n- Sentential negation (e.g., \"not\", \"no\", \"never\")\n- Exclusionary negation (e.g., \"except\", \"besides\")\n- Affixal negation (prefixes like \"un-\", \"dis-\")\n- Implicit negation (verbs or words implying negation like \"deny\", \"refuse\")\n- Contrasting antonyms (pairs of antonyms implying negation)\n\nFor each negation detected, return:\n- The type name from the taxonomy\n- The character span (start and end indices) in the query text\n- A confidence score between 0 and 1 indicating detection certainty\n- A boolean field \"is_negated\" that is true if any negation is detected.\n\n2. Negation Rewriting:\nIf possible, rewrite the query so that no negation or exclusion terms remain, but the original meaning is preserved.\nExample: \"Bread not containing milk\" to \"Dairy-free bread\" or \"Vegan bread\".\n\nIf such rewriting is not possible, explicitly identify the exclusion criteria (terms or phrases to filter out after retrieval).\nExample: \"Cities in Germany except for Berlin\" rewritten_query: \"Cities in Germany\", exclusion_criteria: [\"Berlin\"].\n\n3. Output Format:\nRespond ONLY in JSON format with the following structure:\n{{\n  \"negation_analysis\": [\n    {{\n      \"type\": \"<negation type>\",\n      \"span\": [start_index, end_index],\n      \"confidence\": <float between 0 and 1>\n    }}\n  ],\n  \"is_negated\": <true/false>,\n  \"rewritten_query\": \"<rewritten positive query or original query>\",\n  \"exclusion_criteria\": [\"<list of terms or phrases to exclude>\"],\n  \"explanation\": \"<brief explanation of rewrite/filter decision>\"\n}}",
#     "Data_Path": "train.json",
#     "Input_Field": "query",
#     "Parser_Format": "negation_rewriting_full_parser.py"
# }

print(f"Creating {len(new_nevir_files)} task files for NevIR dataset splits.")
print(f"New NevIR files:")
for f in new_nevir_files:
    print(f"  {f}")

description = "You will be given a natural language query. Your task is to perform a comprehensive negation analysis and rewriting according to the following steps:\n\n1. Negation Detection:\nAnalyze the text in the \"query\" field and detect all types of negation based on the following taxonomy:\n- Sentential negation (e.g., \"not\", \"no\", \"never\")\n- Exclusionary negation (e.g., \"except\", \"besides\")\n- Affixal negation (prefixes like \"un-\", \"dis-\")\n- Implicit negation (verbs or words implying negation like \"deny\", \"refuse\")\n- Contrasting antonyms (pairs of antonyms implying negation)\n\nFor each negation detected, return:\n- The type name from the taxonomy\n- The character span (start and end indices) in the query text\n- A confidence score between 0 and 1 indicating detection certainty\n- A boolean field \"is_negated\" that is true if any negation is detected.\n\n2. Negation Rewriting:\nIf possible, rewrite the query so that no negation or exclusion terms remain, but the original meaning is preserved.\nExample: \"Bread not containing milk\" to \"Dairy-free bread\" or \"Vegan bread\".\n\nIf such rewriting is not possible, explicitly identify the exclusion criteria (terms or phrases to filter out after retrieval).\nExample: \"Cities in Germany except for Berlin\" rewritten_query: \"Cities in Germany\", exclusion_criteria: [\"Berlin\"].\n\n3. Output Format:\nRespond ONLY in JSON format with the following structure:\n{{\n  \"negation_analysis_q1\": [\n    {{\n      \"type\": \"<negation type>\",\n      \"span\": [start_index, end_index],\n      \"confidence\": <float between 0 and 1>\n    }}\n  ],\n  \"is_negated_q1\": <true/false>,\n  \"rewritten_query_q1\": \"<rewritten positive query or original query>\",\n  \"exclusion_criteria_q1\": [\"<list of terms or phrases to exclude>\"],\n  \"explanation_q1\": \"<brief explanation of rewrite/filter decision>\"\n}}"

# create task
i = 0
for file_path in new_nevir_files:
    file_number = 1 if 'q1' in file_path else 2
    description_modified = description.replace('"_q1"', f'"_q2"') if file_number == 2 else description
    task = {
        "Description": description_modified,
        "Data_Path": file_path.replace("data/", ""),
        "Input_Field": f"q{file_number}",
        "Parser_Format": f"negation_rewriting_full_parser_q{file_number}.py"
    }
    task_file = f"tasks/Task{i:03d}.json"

    with open(task_file, "w") as f:
        json.dump(task, f, indent=2)
    
    i += 1

# Copy parser file using Python's shutil
parser_dest_dir = "tasks/parsers"
os.makedirs(parser_dest_dir, exist_ok=True)

# Source file is in the original directory
parser_source_q1 = os.path.join(original_dir, "Parsers", "negation_rewriting_full_parser_q1.py")
parser_dest_q1 = os.path.join(parser_dest_dir, "negation_rewriting_full_parser_q1.py")
parser_source_q2 = os.path.join(original_dir, "Parsers", "negation_rewriting_full_parser_q2.py")
parser_dest_q2 = os.path.join(parser_dest_dir, "negation_rewriting_full_parser_q2.py")

print(f"\nCopying parser file...")
print(f"  Source: {parser_source_q1}")
print(f"  Destination: {os.path.abspath(parser_dest_q1)}")

try:
    if os.path.exists(parser_source_q1):
        shutil.copy2(parser_source_q1, parser_dest_q1)
        shutil.copy2(parser_source_q2, parser_dest_q2)
        print(f"Successfully copied parser files")
    else:
        print(f"ERROR: Source parser file not found: {parser_source_q1}", file=sys.stderr)
        print(f"ERROR: Source parser file not found: {parser_source_q2}", file=sys.stderr)
except Exception as e:
    print(f"Failed to copy parser file: {e}", file=sys.stderr)

print("\n=== Summary ===")
if TESTING_MODE:
    print("TESTING MODE ENABLED - Only 1 entry per dataset processed")
print(f"Current directory: {os.getcwd()}")
print(f"Created {i} task files")
print("\nExtractinate commands to run from NevIR directory:")
for task_num in range(0, i):
    print(f"  extractinate --task_id {task_num:03d} --model_name qwen3:8b --num_predict 4096 --overwrite")
print("\nOutput will be found in: output/run/Task{XXX}-run0/nlp-predictions-dataset.json")



# Now run extractinate --task_id {task_id} --model_name qwen3:8b --num_predict 4096" for each task from within the NevIR folder.

# At the end, the output can be found in NevIR/output/run/Task{task_id}-run0/nlp-predictions-dataset.json

# Or, run process_NevIR_part2.py to collect and rename the outputs automatically.
