import argparse
import json

def compute_stats(json_data):
    category_count = {}

    # Count occurrences of each category_id in annotations
    for annotation in json_data.get("annotations", []):
        cat_id = annotation.get("category_id")
        if cat_id is not None:
            category_count[cat_id] = category_count.get(cat_id, 0) + 1

    # Print name and count of each category
    print("Category statistics:")
    for category in json_data.get("categories", []):
        cat_id = category["id"]
        cat_name = category["name"]
        count = category_count.get(cat_id, 0)
        print(f"- {cat_name}: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute category-wise annotation stats.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file")
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    compute_stats(dataset)
