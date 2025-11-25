import torch
import numpy as np
import pandas as pd
import json

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, features, ranking, img_dataset):
        features_dataset = pd.read_csv(open(features, "r"), sep=';', engine='python')

        # Load triplets and clean columns
        self.triplets = pd.read_csv(open(dataset, "r"), sep=',', engine='python')
        self.triplets.columns = self.triplets.columns.str.strip().str.replace('\ufeff', '')

        # Load image JSON (COCO style)
        dataset_json = json.load(open(img_dataset, "r"))

        # Map file_name to image_id
        file_to_id = {img["file_name"]: img["id"] for img in dataset_json["images"]}

        # ?? Filter valid image IDs with segmentation or bbox
        valid_ann_ids = set()
        for ann in dataset_json["annotations"]:
            has_segmentation = ann.get("segmentation") and len(ann["segmentation"][0]) > 0
            has_bbox = ann.get("bbox") and len(ann["bbox"]) == 4
            if has_segmentation or has_bbox:
                valid_ann_ids.add(ann["image_id"])

        # ?? Filter triplets to only valid IDs
        valid_ids = set(features_dataset["image_id"]).intersection(valid_ann_ids)
        self.triplets = self.triplets[
            self.triplets["case_id"].isin(valid_ids) &
            self.triplets["case_id_pos"].isin(valid_ids) &
            self.triplets["case_id_neg"].isin(valid_ids)
        ].reset_index(drop=True)

        # Extract all unique IDs used in triplets
        triplets_ids = set(
            self.triplets["case_id"].tolist() +
            self.triplets["case_id_pos"].tolist() +
            self.triplets["case_id_neg"].tolist()
        )

        # Parse ranking.csv
        ranking_df = pd.read_csv(open(ranking, "r"), sep=';', engine='python')
        ranking_image_names = set()
        for index, row in ranking_df.iterrows():
            if index < 2:
                continue
            for col in row.index:
                if col not in ['id_casi', 'TIPO DI ULCERA'] and row[col] != '-1':
                    ranking_image_names.add(col)
            if pd.notna(row['id_casi']):
                ranking_image_names.add(row['id_casi'])

        # Convert ranking names to IDs
        ranking_ids = {file_to_id[fname] for fname in ranking_image_names if fname in file_to_id}

        # Final needed image IDs (intersection with annotated)
        needed_ids = triplets_ids.union(ranking_ids).intersection(valid_ann_ids)

        # Filter features_dataset
        features_dataset = features_dataset[features_dataset["image_id"].isin(needed_ids)].reset_index(drop=True)

        # Save image_ids and labels
        self.ids = list(features_dataset["image_id"])
        self.lbls = list(features_dataset["type"])

        # Convert string features to torch tensors
        features = [np.array(eval(f)) for f in features_dataset["feature"]]
        features = np.array(features).squeeze()
        features = torch.from_numpy(features).float()
        self.features = [f.requires_grad_() for f in features]

        # Build image file name ? ID map
        image_names = {}
        for img in dataset_json["images"]:
            if img["id"] in self.ids:
                image_names[img["file_name"]] = img["id"]

        # Build ranking list per image ID
        self.ids_ranking = {}
        image_names_keys = list(image_names.keys())
        for index, row in ranking_df.iterrows():
            row_values = {}
            if index >= 2:
                max_rank = -1
                for col, val in row.items():
                    if (col != 'id_casi' and col != 'TIPO DI ULCERA' and val != '-1' and row['id_casi'] in image_names_keys):
                        row_values[int(val)] = col
                        max_rank = max(max_rank, int(val))
                if row['id_casi'] in image_names_keys:
                    image_id = image_names[row['id_casi']]
                    self.ids_ranking[image_id] = []
                    for i in range(1, max_rank + 1):
                        if row_values.get(i) in image_names:
                            ranked_id = image_names[row_values[i]]
                            self.ids_ranking[image_id].append(ranked_id)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        image_id = self.triplets.iloc[idx]["case_id"]
        anchor = self.features[self.ids.index(image_id)]
        image_id = self.triplets.iloc[idx]["case_id_pos"]
        positive = self.features[self.ids.index(image_id)]
        image_id = self.triplets.iloc[idx]["case_id_neg"]
        negative = self.features[self.ids.index(image_id)]
        return anchor, positive, negative

    def get_ids_ranking(self):
        return self.ids_ranking

    def get_features_dataset(self):
        return self.ids, self.features, self.lbls
