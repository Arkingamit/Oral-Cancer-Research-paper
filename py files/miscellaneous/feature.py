import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
IMG_FOLDER = "/home/21bce072/oral2/datasets/oral1/"  # folder containing all images
COCO_JSON = "/home/21bce072/oral2/datasets/coco_dataset.json"
OUTPUT_CSV = "/home/21bce072/oral2/features_dataset.csv"

# === Load COCO-Style Dataset ===
with open(COCO_JSON, "r") as f:
    coco_data = json.load(f)

images = coco_data["images"]
annotations = coco_data["annotations"]

# === Build ID to Filename & Label Maps ===
id_to_filename = {img["id"]: img["file_name"] for img in images}
id_to_label = {}
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in id_to_label:
        id_to_label[img_id] = ann["category_id"]  # assuming one annotation per image

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet means
        std=[0.229, 0.224, 0.225]    # Imagenet stds
    ),
])

# === Load Pretrained ResNet50 ===
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Feature Extraction ===
records = []
for img_id, file_name in tqdm(id_to_filename.items(), desc="Extracting features"):
    img_path = os.path.join(IMG_FOLDER, file_name)
    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(img_tensor).squeeze().cpu().numpy()
    
    records.append({
        "case_id": img_id,
        "feature": str(feature.tolist()),  # Store list as string
        "type": id_to_label.get(img_id, -1)  # default to -1 if label not found
    })

# === Save to CSV ===
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, sep=';', index=False)
print(f"\n? Saved feature dataset to {OUTPUT_CSV}")
