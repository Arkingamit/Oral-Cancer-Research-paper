import hydra
import torch
import pandas as pd
import json
import numpy as np
from src.metrics import (
    get_knn_classification,
    get_knn,
    log_compound_metrics,
    spearman_footrule_distance,
    kendall_tau_distance
)


from src.models.triplet_net import TripletNetModule
from src.datasets.triplet_dataset import TripletDataset
from src.utils import *

@hydra.main(version_base=None, config_path="./config", config_name="config_projection")
def main(cfg):
    # Load ranking.csv to get reference image file names
    dataset = pd.read_csv(open(cfg.features_extractor.ranking, "r"), sep=';', engine='python')
    references = list(dataset.columns)[2:]

    # Load original JSON to map file names to IDs
    dataset = json.load(open(cfg.features_extractor.original, "r"))
    image_names = {
        image["file_name"]: image["id"]
        for image in dataset["images"]
        if image["file_name"] in references
    }


    references = [ref for ref in references if ref in image_names]
    reference_ids = [image_names[ref] for ref in references]

    # Load dataset
    test_dataset = TripletDataset(
        cfg.dataset.test,
        cfg.triplet.features_dataset,
        cfg.features_extractor.ranking,
        cfg.features_extractor.original
    )

    gt = test_dataset.get_ids_ranking()
    feature_ids, features, lbls = test_dataset.get_features_dataset()

    # Project features using TripletNet if configured
    if cfg.triplet.projection:
        model = TripletNetModule.load_from_checkpoint(cfg.log.path + cfg.triplet.checkpoint_path)
        model.eval().to('cuda')

        predictions = []
        for feature in features:
            feature = feature.to('cuda').detach()
            predictions.append(model.forward(feature))

        predictions = [pred.cpu() for pred in predictions]
    else:
        predictions = [feat.detach().cpu() for feat in features]

    # Process reference features (with skip guard)
    reference_predictions = []
    reference_lbls = []

    for item in reference_ids:
        if item not in feature_ids:
            print(f"id not found skipping.")
            continue
        index = feature_ids.index(item)
        reference_predictions.append(np.array(predictions[index].detach().numpy()))
        reference_lbls.append(lbls[index])

    # Process test features (with skip guard)
    test_predictions = []
    test_lbls = []
    test_ids = list(gt.keys())

    for item in test_ids:
        if item not in feature_ids:
            print(f"id not found skipping.")
            continue
        index = feature_ids.index(item)
        test_predictions.append(np.array(predictions[index].detach().numpy()))
        test_lbls.append(lbls[index])

    # Compute metrics
    k = len(reference_predictions)
    accuracy = get_knn_classification(reference_predictions, reference_lbls, test_predictions, test_lbls, 5)
    app = get_knn(reference_ids, reference_predictions, test_predictions, k)

    # Map test IDs to ranked prediction IDs
    knn = {}
    for i, item in enumerate(test_ids):
        if item not in feature_ids:
            continue
        knn[item] = app[i]

    # Log results
    log_dir = cfg.log.path + cfg.triplet.log_dir
    log_compound_metrics(gt, knn, accuracy, cfg.triplet.projection, log_dir=log_dir)

if __name__ == '__main__':
    main()
