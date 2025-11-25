# import torch
# import numpy as np
# import re
# import pandas as pd
# import json

# class TripletDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, features, ranking, img_dataset):
#         features_dataset = pd.read_csv(open(features, "r"), sep=';', engine='python')

#         # collect the images' id in the triplet dataset
#         self.triplets = pd.read_csv(open(dataset, "r"), sep=',', engine='python')
#         triplets_ids=[]
#         for i in range(0, len(self.triplets)):
#             if self.triplets.loc[i, "case_id"] not in triplets_ids:
#                 triplets_ids.append(self.triplets.loc[i, "case_id"])
#             if self.triplets.loc[i, "case_id_pos"] not in triplets_ids:
#                 triplets_ids.append(self.triplets.loc[i, "case_id_pos"])
#             if self.triplets.loc[i, "case_id_neg"] not in triplets_ids:
#                 triplets_ids.append(self.triplets.loc[i, "case_id_neg"])

#         # mantain only features in the triplet dataset
#         i = 0
#         while i < len(features_dataset):
#             if features_dataset.loc[i, "image_id"] not in triplets_ids:
#                 features_dataset = features_dataset.drop(i)
#             i+=1
#         features_dataset.reset_index(inplace = True, drop = True)

#         self.ids = list(features_dataset["image_id"])

#         features = list(features_dataset["feature"])
#         features=[np.array(eval(feature)) for feature in features]
#         features = np.array(features)
#         features = features.squeeze()
#         features = torch.from_numpy(features)
#         features = [feature.to(torch.float32) for feature in features]
#         self.features = [feature.requires_grad_() for feature in features]

#         self.lbls = list(features_dataset["type"])

#         # create a mapping between image name and id
#         dataset = json.load(open(img_dataset, "r"))
#         image_names={}
#         for image in dataset["images"]:
#             if image["id"] in self.ids:
#                 image_names[image["file_name"]] = image["id"]

#         # create a map containing for each image id, ranked array of the other images' id in the triplet dataset
#         #       1235 1236 1237 1238
#         # 1234    3   -1    1    2
#         #
#         # 1234: [1237, 1238, 1235]

#         self.ids_ranking={}
#         image_names_keys = list(image_names.keys())

#         ranking = pd.read_csv(open(ranking, "r"), sep=';', engine='python')
#         for index, row in ranking.iterrows():
#             row_values={} 
#             if index>=2:
#                 mox = -1
#                 for column, value in row.items():
#                     if (column != 'id_casi' and column != 'TIPO DI ULCERA' and value != '-1' 
#                             and ranking.iloc[index, 0] in list(image_names.keys())):
#                         row_values[int(value)]=column
#                         if mox < int(value):
#                             mox = int(value)
#                 if ranking.iloc[index, 0] in image_names_keys:
#                     image_id = image_names[ranking.iloc[index, 0]]
#                     j = 0
#                     self.ids_ranking[image_id]=[]
#                     for i in range(1, mox+1):
#                         if row_values[i] in image_names:
#                             id_rank = image_names[row_values[i]]
#                             self.ids_ranking[image_id].append(id_rank)
#                             j+=1


#     def __len__(self):
#         return len(self.triplets)

#     def __getitem__(self, idx):
#         image_id = self.triplets.iloc[idx]["case_id"]
#         index = self.ids.index(image_id)
#         anchor = self.features[index]
#         image_id = self.triplets.iloc[idx]["case_id_pos"]
#         index = self.ids.index(image_id)
#         positive = self.features[index]
#         image_id = self.triplets.iloc[idx]["case_id_neg"]
#         index = self.ids.index(image_id)
#         negative = self.features[index]

#         return anchor, positive, negative

#     def get_ids_ranking(self):
#         return self.ids_ranking

#     def get_features_dataset(self):
#         return self.ids, self.features

import torch
import numpy as np
import json
import pandas as pd

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, features, ranking, img_dataset_json):
        # Load features CSV
        features_dataset = pd.read_csv(open(features, "r"), sep=';', engine='python')

        # Load triplets CSV
        self.triplets = pd.read_csv(open(dataset, "r"), sep=',', engine='python')
        triplets_ids = set(self.triplets["case_id"]) | set(self.triplets["case_id_pos"]) | set(self.triplets["case_id_neg"])

        # Filter features dataset to only include IDs present in triplets
        features_dataset = features_dataset[features_dataset["case_id"].isin(triplets_ids)]
        features_dataset.reset_index(inplace=True, drop=True)

        # Save image IDs
        self.ids = list(features_dataset["case_id"])

        # Extract and convert features
        features = [np.array(eval(f)) for f in features_dataset["feature"]]
        features = torch.tensor(np.squeeze(np.stack(features)), dtype=torch.float32)
        self.features = [f.requires_grad_() for f in features]

        # Store class labels
        self.lbls = list(features_dataset["type"])

        # Load the COCO-style JSON image dataset
        with open(img_dataset_json, "r") as f:
            dataset = json.load(f)

               # Map file names to IDs (filter only images present in feature CSV)
        self.image_name_to_id = {
            img["file_name"]: img["id"]
            for img in dataset["images"]
            if img["id"] in self.ids
        }
        
        # Map image IDs to category labels
        self.image_id_to_label = {}
        for ann in dataset["annotations"]:
            image_id = ann["image_id"]
            if image_id in self.ids:
                self.image_id_to_label[image_id] = ann["category_id"]

        # Load ranking CSV and build ID-based ranking
        ranking_df = pd.read_csv(open(ranking, "r"), sep=';', engine='python')
        self.ids_ranking = {}

        for _, row in ranking_df.iterrows():
            base_img = row["id_casi"]
            if base_img not in self.image_name_to_id:
                continue

            image_id = self.image_name_to_id[base_img]
            row_values = {}

            for col, val in row.items():
                if col not in ['id_casi', 'TIPO DI ULCERA'] and val != '-1':
                    try:
                        rank = int(val)
                        row_values[rank] = col
                    except ValueError:
                        continue

            ranked_ids = [
                self.image_name_to_id[file_name]
                for i in sorted(row_values.keys())
                if (file_name := row_values[i]) in self.image_name_to_id
            ]
            self.ids_ranking[image_id] = ranked_ids

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Get triplet image IDs
        anchor_id = self.triplets.iloc[idx]["case_id"]
        pos_id = self.triplets.iloc[idx]["case_id_pos"]
        neg_id = self.triplets.iloc[idx]["case_id_neg"]

        # Retrieve feature vectors
        anchor = self.features[self.ids.index(anchor_id)]
        positive = self.features[self.ids.index(pos_id)]
        negative = self.features[self.ids.index(neg_id)]

        return anchor, positive, negative

    def get_ids_ranking(self):
        return self.ids_ranking

    def get_features_dataset(self):
        return self.ids, self.features
