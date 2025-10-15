import os
import json

def convert_all_jsonl_to_json(dir_path):
    # List all files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.jsonl'):
            jsonl_path = os.path.join(dir_path, filename)
            json_path = os.path.join(dir_path, filename.replace('.jsonl', '.json'))

            # Load each jsonl line as an object
            with open(jsonl_path, 'r', encoding='utf-8') as f_jsonl:
                data = [json.loads(line) for line in f_jsonl]

            # Write out as JSON array
            with open(json_path, 'w', encoding='utf-8') as f_json:
                json.dump(data, f_json, ensure_ascii=False, indent=2)
            print(f"Converted {filename} -> {os.path.basename(json_path)}")

# Example usage:
directory = 'QUEST/data'
convert_all_jsonl_to_json(directory)
