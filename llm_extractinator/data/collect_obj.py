import json
import sys


def main(dataset_path: str, dataset_name: str):
    # Load original JSON file containing your query objects
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    print(f"Total entries in dataset: {len(data)}")

    # Split into two json files based on whether 'rewritten_query' is empty or not
    failed_data = [
        item for item in data if (
            item.get('rewritten_query') == '' and
            item.get('exclusion_criteria') in [None, [], ''] and
            item.get('explanation') in [None, '', []]
        )
    ]


    print(f"Filtered failed entries count: {len(failed_data)}")

    successful_data = [
        item for item in data if not (
            item.get('rewritten_query') == '' and
            item.get('exclusion_criteria') in [None, [], ''] and
            item.get('explanation') in [None, '', []]
        )
    ]

    print(f"Filtered successful entries count: {len(successful_data)}")

    if len(successful_data) == len(data):
        print(f"All entries were successfully processed, saving to path data/{dataset_name}_final.json")
        # Save the successful list to a new JSON file
        with open(f'data/{dataset_name}_final.json', 'w', encoding='utf-8') as outfile:
            json.dump(successful_data, outfile, ensure_ascii=False, indent=2)

    else:
        print(f"saving failed to path data/{dataset_name}_failed.json and successful entries to data/{dataset_name}_successful.json")
        # Save the filtered lists to a new JSON file
        with open(f'data/{dataset_name}_failed.json', 'w', encoding='utf-8') as outfile:
            json.dump(failed_data, outfile, ensure_ascii=False, indent=2)

        with open(f'data/{dataset_name}_successful.json', 'w', encoding='utf-8') as outfile:
            json.dump(successful_data, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # get arguments from command line
    dataset_path = sys.argv[1]
    dataset_name = sys.argv[2]
    main(dataset_path, dataset_name)
    