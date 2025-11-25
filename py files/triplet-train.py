import os
import hydra
import torch
import pytorch_lightning
import time
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from src.models.triplet_net import TripletNetModule
from src.datasets.triplet_datamodule import TripletDataModule
from src.loss_log import LossLogCallback
from src.utils import get_early_stopping, get_loggers


def evaluate_embeddings(model, dataset, phase, seed, time_sec=0):
    model.eval()
    model.freeze()

    all_embeddings = []
    all_labels = []

    ids, features = dataset.get_features_dataset()
    labels_dict = dict(zip(ids, dataset.lbls))

    with torch.no_grad():
        for idx, feature in enumerate(features):
            feature = feature.unsqueeze(0).to(model.device)
            embedding = model(feature).cpu()  # <- Fixed line
            all_embeddings.append(embedding)
            all_labels.append(labels_dict[ids[idx]])

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = np.array(all_labels)

    # Split 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        all_embeddings, all_labels, test_size=0.2, random_state=seed, stratify=all_labels
    )

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    try:
        probs = knn.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, probs, multi_class='ovo')
    except:
        roc_auc = float('nan')

    TP = np.diag(cm).sum()
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    return {
        "Seed": seed,
        "Phase": phase,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-score": report['weighted avg']['f1-score'],
        "Accuracy": (preds == y_test).sum() / len(y_test),
        "ROC_AUC": roc_auc,
        "TP": TP,
        "TN": TN.sum(),
        "FP": FP.sum(),
        "FN": FN.sum(),
        "Time_sec": time_sec
    }


@hydra.main(version_base=None, config_path="./config", config_name="config_projection")
def main(cfg):
    results = []

    for run in range(10):
        print(f"\n----- Run {run+1}/10 -----")

        # Generate a new random seed if -1
        seed = torch.randint(0, 10000, (1,)).item() if cfg.train.seed == -1 else cfg.train.seed
        torch.manual_seed(seed)

        # Callbacks & Logging
        callbacks = [
            get_early_stopping(cfg),
            LossLogCallback(cfg),
            ModelCheckpoint(
                dirpath=f"./checkpoints/run_{run+1}",
                filename="best_model",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
            )
        ]
        loggers = get_loggers(cfg)

        # Initialize model
        model = TripletNetModule(
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )

        # Load Data
        data = TripletDataModule(
            train=cfg.dataset.train,
            val=cfg.dataset.val,
            test=cfg.dataset.test,
            features=cfg.features_extractor.features_dataset,
            ranking=cfg.features_extractor.ranking,
            img_dataset=cfg.features_extractor.original,
            batch_size=cfg.train.batch_size
        )

        # Trainer
        trainer = pytorch_lightning.Trainer(
            logger=loggers,
            callbacks=callbacks,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            log_every_n_steps=1,
            max_epochs=cfg.train.max_epochs
        )

        # Train
        start_time = time.time()
        trainer.fit(model, datamodule=data)
        train_time = time.time() - start_time

        # Evaluate on embeddings
        print(f"[Run {run+1}] Evaluating embeddings...")
        train_metrics = evaluate_embeddings(model, data.train_dataset, phase="Train", seed=seed, time_sec=train_time)
        test_metrics = evaluate_embeddings(model, data.get_test_dataset(), phase="Test", seed=seed)

        results.append(train_metrics)
        results.append(test_metrics)

        # Save model checkpoint
        model_path = f"./checkpoints/run_{run+1}/final_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"[Run {run+1}] Saved model to {model_path}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("training_test_metrics.csv", index=False)
    print("\n? All runs completed. Metrics saved to training_test_metrics.csv")


if __name__ == "__main__":
    main()
