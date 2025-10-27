import json

# Load input JSON file
with open('data/NevIR_validation_q1_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for obj in data:
    # Rename fields if present
    if 'query' in obj:
        obj['q1'] = obj.pop('query')
    if 'document' in obj:
        obj['doc1'] = obj.pop('document')

# Save to output file
with open('data/NevIR_validation_q1_final_renamed.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("Field renaming completed: 'NevIR_validation_q1_final_renamed.json' created.")