import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# === Paths ===
json_path = "/home/21bce072/oral2/datasets/coco_dataset.json"         # COCO JSON path
image_root = "datasets/oral1"           # Path to image folder
output_root = "reddish_output"           # Output directory
report_path = "reddish_lesion_report.csv"

# === Load data ===
with open(json_path) as f:
    data = json.load(f)

image_id_map = {img['id']: img for img in data['images']}

# === Config ===
R_THRESHOLD = 130         # Minimum red channel value
RED_DOMINANCE = 1.3       # Red should be 1.3x greater than G & B

os.makedirs(output_root, exist_ok=True)

category_red_counts = defaultdict(int)
category_total_counts = defaultdict(int)
results = []

def is_reddish_region(region):
    b, g, r = cv2.split(region)
    red_mask = (r > R_THRESHOLD) & (r > RED_DOMINANCE * g) & (r > RED_DOMINANCE * b)
    ratio = np.sum(red_mask) / (region.shape[0] * region.shape[1])
    return ratio > 0.10  # Consider reddish if >10% pixels match

# === Process annotations ===
for ann in tqdm(data['annotations']):
    img_info = image_id_map.get(ann['image_id'])
    if not img_info or not ann['segmentation']:
        continue

    category_id = ann['category_id']
    category_total_counts[category_id] += 1

    img_path = os.path.join(image_root, img_info['file_name'])
    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Create lesion mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in ann['segmentation']:
        pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

    x, y, bw, bh = map(int, ann['bbox'])
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(bbox_mask, (x, y), (x + bw, y + bh), 255, -1)

    reddish = is_reddish_region(cv2.bitwise_and(image, image, mask=bbox_mask))

    # Save image if reddish detected
    if reddish:
        category_red_counts[category_id] += 1
        cat_dir = os.path.join(output_root, f"category_{category_id}")
        os.makedirs(cat_dir, exist_ok=True)
        out_path = os.path.join(cat_dir, img_info['file_name'])
        cv2.imwrite(out_path, image)

    results.append({
        "image_id": ann['image_id'],
        "category_id": category_id,
        "reddish_bbox": reddish,
        "bbox": ann['bbox'],
        "area": ann['area']
    })

# === Save detailed report ===
df = pd.DataFrame(results)
df.to_csv(report_path, index=False)
print(f"Reddish lesion report saved to: {report_path}")

# === Save summary per category ===
summary = []
for cat_id in sorted(category_total_counts.keys()):
    total = category_total_counts[cat_id]
    red = category_red_counts.get(cat_id, 0)
    summary.append({
        "category_id": cat_id,
        "total_images": total,
        "reddish_detected": red,
        "percentage": (red / total) * 100 if total > 0 else 0
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("reddish_summary_per_category.csv", index=False)
print("Summary per category saved to reddish_summary_per_category.csv")
