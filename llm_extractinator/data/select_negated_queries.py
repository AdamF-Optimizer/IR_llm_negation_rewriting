import json
import math

def clean_nan(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def clean_object(obj):
    # Recursively clean object to replace NaN with None
    if isinstance(obj, dict):
        return {k: clean_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_object(i) for i in obj]
    else:
        return clean_nan(obj)

def filter_negated_queries(input_file, output_file):
    # Read input JSON file
    with open(input_file, "r") as infile:
        data = json.load(infile)

    # Filter for objects where is_negated is True
    filtered_data = []
    for obj in data:
        if obj.get("is_negated") is True:
            cleaned_obj = clean_object(obj)
            filtered_data.append(cleaned_obj)


    # Write filtered objects to output JSON file
    with open(output_file, "w") as outfile:
        json.dump(filtered_data, outfile, indent=2)


filter_negated_queries("QUEST/data/val_llm_processed.json", "QUEST/data/val_negations.json")
filter_negated_queries("QUEST/data/train_llm_processed.json", "QUEST/data/train_negations.json")
filter_negated_queries("QUEST/data/test_llm_processed.json", "QUEST/data/test_negations.json")