import json

# Load JSON files
with open('data/NevIR_validation_q1_final.json', 'r', encoding='utf-8') as f1, open('data/NevIR_validation_q2_final.json', 'r', encoding='utf-8') as f2:
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

# Write merged output
with open('data/NevIR_validation_final.json', 'w') as out:
    json.dump(merged, out, indent=2)
print("Merged JSON files into 'data/NevIR_validation_final.json'")