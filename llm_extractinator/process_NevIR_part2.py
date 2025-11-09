import os
import shutil
import json

nevir_files = ['NevIR/data/NevIR_train_q1.json', 'NevIR/data/NevIR_train_q2.json',
                'NevIR/data/NevIR_test_q1.json', 'NevIR/data/NevIR_test_q2.json',
                'NevIR/data/NevIR_validation_q1.json', 'NevIR/data/NevIR_validation_q2.json']

# collect results and rename:
for task_id in range(0, 6, 2):
    print(f"Processing Task {task_id}")
    output_file_q1 = f"NevIR/output/run/Task{task_id:03}-run0/nlp-predictions-dataset.json"
    output_file_q2 = f"NevIR/output/run/Task{task_id+1:03}-run0/nlp-predictions-dataset.json"
    if os.path.exists(output_file_q1) and os.path.exists(output_file_q2):
        # Rename the output files
        new_output_file_q1 = f"{nevir_files[task_id].replace('.json', '')}_negation_processed.json"
        new_output_file_q2 = f"{nevir_files[task_id+1].replace('.json', '')}_negation_processed.json"
        print(f"Copying {output_file_q1} to {new_output_file_q1}")
        shutil.copy2(output_file_q1, new_output_file_q1)
        print(f"Copying {output_file_q2} to {new_output_file_q2}")
        shutil.copy2(output_file_q2, new_output_file_q2)

        combine_output_file = f"{nevir_files[task_id//2].replace('_q1.json', '')}_negation_processed_combined.json"
        print(f"Combining {new_output_file_q1} and {new_output_file_q2} into {combine_output_file}")
        
        # Load JSON files
        with open(new_output_file_q1, 'r', encoding='utf-8') as f1, open(new_output_file_q2, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        # Index by (id, WorkerId)
        map1 = { (d["id"], d["WorkerId"]): d for d in data1 }
        map2 = { (d["id"], d["WorkerId"]): d for d in data2 }

        # Find all keys in either file
        all_keys = set(map1.keys()) | set(map2.keys())

        merged = []
        for key in all_keys:
            if key in map1 and key in map2:
                # Merge both objects into one, combining all fields
                combined = {**map1[key], **map2[key]}
                merged.append(combined)
            elif key in map1:
                merged.append(map1[key])
            else:
                merged.append(map2[key])


        print(f"Merged total entries: {len(merged)}")

        # Write merged output
        with open(new_output_file_q1.replace('_q1_negation_processed.json', '_negation_processed_combined.json'), 'w') as out:
            json.dump(merged, out, indent=2)
        print(f"Merged JSON files into '{new_output_file_q1.replace('_q1_negation_processed.json', '_negation_processed_combined.json')}'")
    else:
        print(f"Output file not found for Task {task_id} or Task {task_id+1}")