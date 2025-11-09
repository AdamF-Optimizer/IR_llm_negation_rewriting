import os
import shutil

QUEST_files = ["QUEST/data/QUEST_train.json", "QUEST/data/QUEST_val.json", "QUEST/data/QUEST_test.json"]

# collect results and rename:
for task_id in range(0, 3):
    print(f"Processing Task {task_id}")
    output_file = f"QUEST/output/run/Task{task_id:03}-run0/nlp-predictions-dataset.json"
    if os.path.exists(output_file):
        # Rename or move the output file as needed
        new_output_file = f"{QUEST_files[task_id].replace('.json', '')}_negation_processed.json"
        print(f"Copying {output_file} to {new_output_file}")
        shutil.copy2(output_file, new_output_file)
    else:
        print(f"Output file not found for Task {task_id}")