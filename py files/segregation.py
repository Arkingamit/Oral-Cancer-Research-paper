import os
import json
from collections import defaultdict
import shutil

# Paths
json_path = "/home/21bce072/oral2/datasets/coco_dataset.json"
image_root = "datasets/oral1"
output_root = "segregated_categories"
os.makedirs(output_root, exist_ok=True)

# Load JSON
data = None
with open(json_path) as f:
    data = json.load(f)

# Create mapping from image_id to image file name
image_map = {img['id']: img['file_name'] for img in data['images']}

# Group image_ids by category
category_image_map = defaultdict(set)
for ann in data['annotations']:
    category_id = ann['category_id']
    image_id = ann['image_id']
    category_image_map[category_id].add(image_id)

# Copy images to category-wise folders
for category_id, image_ids in category_image_map.items():
    category_dir = os.path.join(output_root, f"category_{category_id}")
    os.makedirs(category_dir, exist_ok=True)
    for img_id in image_ids:
        file_name = image_map.get(img_id)
        if file_name:
            src_path = os.path.join(image_root, file_name)
            dst_path = os.path.join(category_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

print("Images segregated by category in:", output_root)
