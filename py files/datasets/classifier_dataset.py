# # import torch
# # import torchvision.transforms as transforms
# # import torchvision.transforms.functional as TF
# # import os
# # import json
# # from PIL import Image
# # import cv2

# # class classifier(torch.utils.data.Dataset):
# #     def __init__(self, annonations, transform=None):
# #         self.annonations = annonations
# #         self.transform = transform

# #         with open(annonations, "r") as f:
# #             self.dataset = json.load(f)
        
# #         self.images = dict()
# #         for image in self.dataset["images"]:
# #             self.images[image["id"]] = image
        
# #         self.categories = dict()
# #         for i, category in enumerate(self.dataset["categories"]):
# #             self.categories[category["id"]] = i

        
# #     def __len__(self):
# #         return len(self.dataset["annotations"])

# #     def __getitem__(self, idx):
# #         annotation = self.dataset["annotations"][idx]
# #         image = self.images[annotation["image_id"]]
# #         image_path = os.path.join(os.path.dirname(self.annonations), "oral1", image["file_name"])
# #         image = Image.open(image_path).convert("RGB")
        
# #         x, y, w, h = annotation["bbox"]
# #         subimage = image.crop((x, y, x+w, y+h))

# #         if self.transform:
# #             subimage = self.transform(subimage)

# #         category = self.categories[annotation["category_id"]]

# #         return subimage, category
    
# #     def get_image_id(self, idx):
# #         return self.dataset["images"][idx]["id"]

# #     '''def __getitem__(self, idx):
# #         annotation = self.dataset["annotations"][idx]
# #         image = self.images[annotation["image_id"]]
# #         image_path = os.path.join(os.path.dirname(self.annonations), "oral1", image["file_name"])
        
# #         image = cv2.imread(image_path)
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# #         x, y, w, h = annotation["bbox"]
# #         x = int(x)
# #         y = int(y)
# #         w = int(w)
# #         h = int(h)
# #         image = image[y:y+h, x:x+w]

# #         if self.transform:
# #             augmented = self.transform(image=image) 
# #             image = augmented['image']

# #         category = self.categories[annotation["category_id"]]

# #         return image, category
# #     '''

# # if __name__ == "__main__":
# #     import torchvision

# #     dataset = classifier(
# #         "datasets/oral1/train.json",
# #         transform=transforms.Compose([
# #             transforms.Resize((224, 224), antialias=True),
# #             transforms.ToTensor()
# #         ])
# #     )

# #     torchvision.utils.save_image(dataset[1][0], "test.png")

# import os
# import json
# from torch.utils.data import Dataset
# from PIL import Image

# class ClassifierDataset(Dataset):
#     def __init__(self, annotations_path, transform=None):
#         """
#         annotations_path: Path to JSON file containing annotations.
#         JSON structure may have a top-level "object" wrapper or be in COCO-style directly.
#         """
#         self.annotations_path = annotations_path
#         self.transform = transform

#         with open(annotations_path, "r") as f:
#             raw = json.load(f)
#         # support wrapper
#         data = raw.get("object", raw)

#         # read lists
#         self.images_list = data.get("images", [])
#         self.annotations_list = data.get("annotations", [])
#         self.categories_list = data.get("categories", [])

#         # build lookup dicts
#         self.images = {img["id"]: img for img in self.images_list}
#         self.categories = {cat["id"]: idx for idx, cat in enumerate(self.categories_list)}

#     def __len__(self):
#         return len(self.annotations_list)

#     def __getitem__(self, idx):
#         ann = self.annotations_list[idx]
#         img_meta = self.images.get(ann["image_id"])
#         if img_meta is None:
#             raise KeyError(f"Image id {ann['image_id']} not found in annotations")

#         # Determine image file path
#         # Prefer explicit "path" in JSON if available, else construct from file_name
#         img_path = img_meta.get("path") or img_meta.get("file_name")
#         # handle absolute/relative
#         if img_path.startswith("/"):
#             img_path = img_path.lstrip("/")
#         full_path = os.path.join(os.path.dirname(self.annotations_path), img_path)

#         image = Image.open(full_path).convert("RGB")

#         # Crop using bbox
#         x, y, w, h = ann.get("bbox", [0, 0, image.width, image.height])
#         subimage = image.crop((x, y, x + w, y + h))

#         if self.transform:
#             subimage = self.transform(subimage)

#         category_id = ann.get("category_id")
#         label = self.categories.get(category_id)
#         if label is None:
#             raise KeyError(f"Category id {category_id} not found in categories")

#         return subimage, label

#     def get_image_id(self, idx):
#         return self.annotations_list[idx].get("image_id")


# if __name__ == "__main__":
#     import torchvision.transforms as T
#     import torchvision

#     ds = ClassifierDataset(
#         "datasets/train.json",
#         transform=T.Compose([
#             T.Resize((224, 224), antialias=True),
#             T.ToTensor()
#         ])
#     )
#     # test fetch
#     img, lbl = ds[0]
#     print(f"Sample label: {lbl}, image size: {img.shape}")
#     torchvision.utils.save_image(img, "test.png")
import os
import hydra
import json
from torch.utils.data import Dataset
from PIL import Image


class ClassifierDataset(Dataset):
    def __init__(self, annotations_path, transform=None):
        self.annotations_path = annotations_path
        self.transform = transform

        with open(annotations_path, "r") as f:
            raw = json.load(f)

        data = raw.get("object", raw)

        self.images_list = data.get("images", [])
        self.annotations_list = data.get("annotations", [])
        self.categories_list = data.get("categories", [])

        self.images = {img["id"]: img for img in self.images_list}
        self.categories = {cat["id"]: idx for idx, cat in enumerate(self.categories_list)}

        self.valid_data = []
        self.skipped_count = 0

        base_dir = os.path.dirname(self.annotations_path)

        for ann in self.annotations_list:
            img_meta = self.images.get(ann["image_id"])
            if img_meta is None:
                self.skipped_count += 1
                continue

            img_path = img_meta.get("path") or img_meta.get("file_name")
            img_path = img_path.replace("/", os.sep)

            if os.path.isabs(img_path) or img_meta.get("path", "").startswith("/"):
                norm_path = img_path.lstrip(os.sep)
                full_path = norm_path
            else:
                full_path = os.path.join(base_dir, img_path)

            if not os.path.exists(full_path):
                self.skipped_count += 1
                continue

            self.valid_data.append((ann, full_path))

        print(f"[INFO] Loaded {len(self.valid_data)} samples, skipped {self.skipped_count} missing files.")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        ann, full_path = self.valid_data[idx]

        image = Image.open(full_path).convert("RGB")

        x, y, w, h = ann.get("bbox", [0, 0, image.width, image.height])
        subimage = image.crop((x, y, x + w, y + h))

        if self.transform:
            subimage = self.transform(subimage)

        category_id = ann.get("category_id")
        label = self.categories.get(category_id)
        if label is None:
            raise KeyError(f"Category id {category_id} not found in categories")

        return subimage, label

    def get_image_id(self, idx):
        return self.valid_data[idx][0].get("image_id")


if __name__ == "__main__":
    import torchvision.transforms as T
    import torchvision

    ds = ClassifierDataset(
        "datasets/oral1/train.json",
        transform=T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ToTensor()
        ])
    )

    img, lbl = ds[0]
    print(f"Sample label: {lbl}, image size: {img.shape}")
    torchvision.utils.save_image(img, "test.png")
