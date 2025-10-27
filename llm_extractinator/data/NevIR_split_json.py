import json

nevir_files = ['data/NevIR_train.json', 'data/NevIR_test.json', 'data/NevIR_validation.json']

for nevir_file in nevir_files:
    # Load input file
    with open(nevir_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
