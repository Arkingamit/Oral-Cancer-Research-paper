import os
import hydra
import torch
import pytorch_lightning
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.classifier import OralClassifierModule
from src.log import LossLogCallback, get_loggers
from src.datasets.classifier_datamodule import ClassificationDataModule
from src.utils import *


@hydra.main(version_base=None, config_path="./config", config_name="config_classification")
def main(cfg):
    all_reports = []

    for run_idx in range(10):
        print(f"\n?? Run {run_idx + 1}/10")

        # Generate a unique random seed for every run
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
        torch.manual_seed(cfg.train.seed)

        callbacks = [
            get_early_stopping(cfg),
            LossLogCallback()
        ]

        loggers = get_loggers(cfg)
        torch.set_float32_matmul_precision("high")

        model = OralClassifierModule(
            model=cfg.model.name,
            weights=cfg.model.weights,
            num_classes=cfg.model.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs,
            features_size=cfg.model.features_size,
            frozen_layers=cfg.train.frozen_layers
        )

        train_img_transform, val_img_transform, test_img_transform, img_transform = get_transformations(cfg)
        data = ClassificationDataModule(
            train=cfg.dataset.train,
            val=cfg.dataset.val,
            test=cfg.dataset.test,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            train_transform=train_img_transform,
            val_transform=val_img_transform,
            test_transform=test_img_transform,
            transform=img_transform,
        )

        run_dir = os.path.join(cfg.log.path, f"{cfg.log.dir}_run{run_idx + 1}")
        os.makedirs(run_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=run_dir,
            filename="best-checkpoint",
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        callbacks.append(checkpoint_callback)

        trainer = pytorch_lightning.Trainer(
            logger=loggers,
            callbacks=callbacks,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            log_every_n_steps=1,
            max_epochs=cfg.train.max_epochs,
        )
        trainer.fit(model, data)
        trainer.test(dataloaders=data.test_dataloader(), ckpt_path='best')

        predictions = trainer.predict(model, data)
        predictions = torch.cat(predictions, dim=0)
        predictions = torch.argmax(predictions, dim=1)
        gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)

        report_dict = classification_report(gt, predictions, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_excel(os.path.join(run_dir, f"classification_report_run{run_idx + 1}.xlsx"))

        macro_avg = report_dict.get("macro avg", {})
        macro_avg["run"] = run_idx + 1
        macro_avg["seed"] = seed  # Save the seed used for this run
        all_reports.append(macro_avg)

    summary_df = pd.DataFrame(all_reports)
    summary_path = os.path.join(cfg.log.path, f"{cfg.log.dir}_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    avg_metrics = summary_df.drop(columns=["run", "seed"]).mean()
    print("\n?? Average Results over 10 Runs (Macro Avg):")
    print(avg_metrics)


if __name__ == "__main__":
    main()
