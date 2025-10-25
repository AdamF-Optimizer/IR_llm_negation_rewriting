import json
import sys


import json

def merge_json_files(file1_path, file2_path, output_path):
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    combined_data = data1 + data2

    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(combined_data, out, ensure_ascii=False, indent=2)

    print(f"Merged file: {file1_path} with {len(data1)} entries and file {file2_path} with {len(data2)} entries into file {output_path} with {len(combined_data)} total entries.")


if __name__ == "__main__":
    # get arguments from command line
    dataset_name = sys.argv[1]
    task = sys.argv[2]
    merge_json_files(f'data/{dataset_name}_successful.json', f'output/run/{task}/nlp-predictions-dataset.json', f'data/{dataset_name}_final.json')
    