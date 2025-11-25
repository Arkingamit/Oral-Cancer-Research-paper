import os
import json
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

# Load JSON
data_path = "/home/21bce072/oral2/datasets/coco_dataset.json"
with open(data_path) as f:
    data = json.load(f)

# Setup paths
image_root = "datasets/oral1"  # change to actual image folder
output_dir = "logs"
os.makedirs(output_dir, exist_ok=True)

# Index by image ID
image_id_map = {img['id']: img for img in data['images']}
annotations_by_cat = defaultdict(list)

# Group annotations by category
for ann in data['annotations']:
    annotations_by_cat[ann['category_id']].append(ann)

# Helper: Calculate circularity of a contour
def calculate_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)

# Save CSV result
circular_report = []

# Use only Canny filter
filter_name = "Canny"
def filter_fn(gray):
    return cv2.Canny(gray, 50, 150)

# Process per category
for category_id, anns in tqdm(annotations_by_cat.items()):
    base_dir = os.path.join(output_dir, filter_name, f"category_{category_id}")
    bbox_dir = os.path.join(base_dir, "bbox_only")
    segm_dir = os.path.join(base_dir, "segmentation_only")
    both_dir = os.path.join(base_dir, "combined")
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(segm_dir, exist_ok=True)
    os.makedirs(both_dir, exist_ok=True)

    for ann in anns:
        image_info = image_id_map[ann['image_id']]
        image_path = os.path.join(image_root, image_info['file_name'])
        image = cv2.imread(image_path)
        if image is None:
            continue

        # === Apply Canny filter on raw image ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = filter_fn(gray)

        # Use filtered image for bounding box overlay
        img_bbox = filtered.copy()
        img_bbox = cv2.cvtColor(img_bbox, cv2.COLOR_GRAY2BGR)

        # Original image for segmentation and combined
        img_segm = image.copy()
        img_comb = image.copy()

        # Segmentation mask from JSON
        h, w = image.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        if ann['segmentation']:
            for seg in ann['segmentation']:
                pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(seg_mask, [pts], 255)

        # Detect contours on segmentation mask
        contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x, y, bw, bh = map(int, ann['bbox'])
        cv2.rectangle(img_bbox, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.rectangle(img_comb, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        circular_detected = False
        for cnt in contours:
            circ = calculate_circularity(cnt)
            if circ >= 0.85:
                circular_detected = True
            cv2.drawContours(img_segm, [cnt], -1, (0, 255, 0), 2)
            cv2.drawContours(img_comb, [cnt], -1, (255, 0, 0), 2)

        fname = image_info['file_name']
        cv2.imwrite(os.path.join(bbox_dir, fname), img_bbox)
        cv2.imwrite(os.path.join(segm_dir, fname), img_segm)
        cv2.imwrite(os.path.join(both_dir, fname), img_comb)

        circular_report.append({
            "image_id": ann['image_id'],
            "category_id": category_id,
            "filter": filter_name,
            "circular_lesion": circular_detected,
            "bbox_area": ann['area'],
            "segmentation_pts": len(ann['segmentation'][0]) if ann['segmentation'] else 0
        })

# Save report
report_df = pd.DataFrame(circular_report)
report_df.to_csv("circular_lesion_report.csv", index=False)
print("Report saved as circular_lesion_report.csv")
