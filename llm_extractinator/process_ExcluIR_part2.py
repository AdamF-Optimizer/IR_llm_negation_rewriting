import os
import shutil

ExcluIR_files = ["ExcluIR/data/test_manual_final.json"]

# collect results and rename:
print(f"Processing Task 0")
output_file = f"ExcluIR/output/run/Task{0:03}-run0/nlp-predictions-dataset.json"
if os.path.exists(output_file):
    # Rename or move the output file as needed
    new_output_file = f"{ExcluIR_files[0].replace('.json', '')}_negation_processed.json"
    print(f"Copying {output_file} to {new_output_file}")
    shutil.copy2(output_file, new_output_file)
else:
    print(f"Output file not found for Task 0")