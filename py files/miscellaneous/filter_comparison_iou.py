import cv2
import json
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm

# Input paths
json_path ='/home/21bce072/oral2/datasets/coco_dataset.json'
image_folder = "/datasets/oral1"
output_folder = "filtered_outputs"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
with open(json_path) as f:
    dataset = json.load(f)

images_by_id = {img["id"]: img for img in dataset["images"]}
annotations_by_image = {}
for ann in dataset["annotations"]:
    if ann["image_id"] not in annotations_by_image:
        annotations_by_image[ann["image_id"]] = []
    annotations_by_image[ann["image_id"]].append(ann)

results = []

def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def polygon_to_mask(segmentation, height, width):
    """Convert COCO polygon segmentation to a binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentation:
        pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask

# Loop over images
for image in tqdm(dataset["images"]):
    img_id = image["id"]
    file_name = image["file_name"]
    img_path = os.path.join(image_folder, file_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    height, width = img.shape[:2]
    anns = annotations_by_image.get(img_id, [])

    for idx, ann in enumerate(anns):
        if "bbox" not in ann or "segmentation" not in ann:
            continue

        x, y, w, h = list(map(int, ann["bbox"]))
        crop = img[y:y+h, x:x+w]
        mask = polygon_to_mask(ann["segmentation"], height, width)[y:y+h, x:x+w]

        # Apply filters
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

        # Binarize all for comparison
        bin_canny = (canny > 0).astype(np.uint8)
        bin_binary = (binary > 0).astype(np.uint8)
        bin_clahe = (clahe > 127).astype(np.uint8)

        # Resize mask if necessary
        if mask.shape != bin_canny.shape:
            mask = cv2.resize(mask, (bin_canny.shape[1], bin_canny.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Compute IoU
        iou_canny = compute_iou(mask, bin_canny)
        iou_binary = compute_iou(mask, bin_binary)
        iou_clahe = compute_iou(mask, bin_clahe)

        # Decide best filter
        ious = {"canny": iou_canny, "binary": iou_binary, "clahe": iou_clahe}
        best_filter = max(ious, key=ious.get)

        results.append({
            "image_id": img_id,
            "file_name": file_name,
            "bbox_index": idx,
            "bbox": ann["bbox"],
            "iou_canny": iou_canny,
            "iou_binary": iou_binary,
            "iou_clahe": iou_clahe,
            "best_filter": best_filter
        })

        # Save visual outputs (optional)
        basename = os.path.splitext(file_name)[0]
        cv2.imwrite(os.path.join(output_folder, f"{basename}_bbox{idx}_canny.jpg"), canny)
        cv2.imwrite(os.path.join(output_folder, f"{basename}_bbox{idx}_binary.jpg"), binary)
        cv2.imwrite(os.path.join(output_folder, f"{basename}_bbox{idx}_clahe.jpg"), clahe)

# Save report
df = pd.DataFrame(results)
df.to_excel("filter_results.xlsx", index=False)
print("? Completed. Report saved to filter_results.xlsx")
