import os
import hydra
import torch
import pytorch_lightning
import numpy as np
import pandas as pd
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.classifier import OralClassifierModule
from src.log import LossLogCallback, get_loggers
from src.datasets.classifier_datamodule import ClassificationDataModule
from src.utils import *

@hydra.main(version_base=None, config_path="./config", config_name="config_classification")
def main(cfg):
    all_detailed_results = []
    all_reports = []

    for run_idx in range(10):
        print(f"\n?? Run {run_idx + 1}/10")

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

        print("?? Starting training phase...")
        train_start_time = time.time()
        trainer.fit(model, data)
        train_end_time = time.time()
        train_time = train_end_time - train_start_time

        # ---------------- TRAINING METRICS CALCULATION ---------------- #
        print("?? Evaluating on training data for metrics...")
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        train_preds, train_targets, train_probs = [], [], []
        with torch.no_grad():
            for batch in data.train_dataloader():
                x, y = batch
                x = x.to(model.device)
                y = y.to(model.device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                train_preds.append(preds.cpu())
                train_targets.append(y.cpu())
                train_probs.append(probs.cpu())

        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_probs = torch.cat(train_probs).numpy()

        train_cm = confusion_matrix(train_targets, train_preds)
        train_report_dict = classification_report(train_targets, train_preds, output_dict=True)

        # ROC-AUC for training
        try:
            if cfg.model.num_classes == 2:
                roc_auc_train = roc_auc_score(train_targets, train_probs[:, 1])
            else:
                gt_binarized = label_binarize(train_targets, classes=range(cfg.model.num_classes))
                roc_auc_train = roc_auc_score(gt_binarized, train_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not calculate training ROC-AUC: {e}")
            roc_auc_train = 0.0

        # Compute TP, TN, FP, FN for training
        if cfg.model.num_classes == 2:
            tn_train, fp_train, fn_train, tp_train = train_cm.ravel()
        else:
            tp_train = np.diag(train_cm).sum()
            tn_train = train_cm.sum() - (train_cm.sum(axis=1) + train_cm.sum(axis=0) - np.diag(train_cm)).sum()
            fp_train = train_cm.sum(axis=0) - np.diag(train_cm)
            fn_train = train_cm.sum(axis=1) - np.diag(train_cm)
            fp_train = fp_train.sum()
            fn_train = fn_train.sum()

        train_metrics = {
            'Iteration': f'Run_{run_idx + 1}',
            'Seed': seed,
            'Precision': train_report_dict.get('macro avg', {}).get('precision', 0.0),
            'Recall': train_report_dict.get('macro avg', {}).get('recall', 0.0),
            'F1-score': train_report_dict.get('macro avg', {}).get('f1-score', 0.0),
            'Accuracy': train_report_dict.get('accuracy', 0.0),
            'Phase': 'Train',
            'TP': tp_train,
            'TN': tn_train,
            'FP': fp_train,
            'FN': fn_train,
            'Time_sec': round(train_time, 2),
            'ROC_AUC': round(roc_auc_train, 4)
        }
        all_detailed_results.append(train_metrics)

        train_report_df = pd.DataFrame(train_report_dict).transpose()
        train_report_df.to_excel(os.path.join(run_dir, f"classification_report_train_run{run_idx + 1}.xlsx"))
        pd.DataFrame(train_cm).to_excel(os.path.join(run_dir, f"confusion_matrix_train_run{run_idx + 1}.xlsx"))

        # ---------------- TESTING PHASE ---------------- #
        print("?? Starting testing phase...")
        test_start_time = time.time()
        trainer.test(dataloaders=data.test_dataloader(), ckpt_path='best')
        test_end_time = time.time()
        test_time = test_end_time - test_start_time

        # Predictions
        predictions = trainer.predict(model, data)
        predictions = torch.cat(predictions, dim=0)
        predictions_proba = torch.softmax(predictions, dim=1)
        predictions = torch.argmax(predictions, dim=1)
        gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)

        predictions_np = predictions.cpu().numpy()
        gt_np = gt.cpu().numpy()
        predictions_proba_np = predictions_proba.cpu().numpy()

        report_dict = classification_report(gt_np, predictions_np, output_dict=True)
        cm = confusion_matrix(gt_np, predictions_np)

        # ROC-AUC
        try:
            if cfg.model.num_classes == 2:
                roc_auc = roc_auc_score(gt_np, predictions_proba_np[:, 1])
            else:
                gt_binarized = label_binarize(gt_np, classes=range(cfg.model.num_classes))
                roc_auc = roc_auc_score(gt_binarized, predictions_proba_np, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")
            roc_auc = 0.0

        if cfg.model.num_classes == 2:
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = np.diag(cm).sum()
            tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)).sum()
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            fp = fp.sum()
            fn = fn.sum()

        test_metrics = {
            'Iteration': f'Run_{run_idx + 1}',
            'Seed': seed,
            'Precision': report_dict.get('macro avg', {}).get('precision', 0.0),
            'Recall': report_dict.get('macro avg', {}).get('recall', 0.0),
            'F1-score': report_dict.get('macro avg', {}).get('f1-score', 0.0),
            'Accuracy': report_dict.get('accuracy', 0.0),
            'Phase': 'Test',
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Time_sec': round(test_time, 2),
            'ROC_AUC': round(roc_auc, 4)
        }
        all_detailed_results.append(test_metrics)

        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_excel(os.path.join(run_dir, f"classification_report_run{run_idx + 1}.xlsx"))
        pd.DataFrame(cm).to_excel(os.path.join(run_dir, f"confusion_matrix_run{run_idx + 1}.xlsx"))

        macro_avg = report_dict.get("macro avg", {})
        macro_avg["run"] = run_idx + 1
        macro_avg["seed"] = seed
        macro_avg["train_time"] = train_time
        macro_avg["test_time"] = test_time
        macro_avg["total_time"] = train_time + test_time
        macro_avg["roc_auc"] = roc_auc
        all_reports.append(macro_avg)

        print(f"? Run {run_idx + 1} completed - Train: {train_time:.2f}s | Test: {test_time:.2f}s")

    detailed_df = pd.DataFrame(all_detailed_results)
    detailed_csv_path = os.path.join(cfg.log.path, f"{cfg.log.dir}_detailed_results.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    print(f"?? Detailed results saved to: {detailed_csv_path}")

    summary_df = pd.DataFrame(all_reports)
    summary_path = os.path.join(cfg.log.path, f"{cfg.log.dir}_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)

    avg_metrics = summary_df.drop(columns=["run", "seed"]).mean()
    print("\n?? Average Results over 10 Runs (Macro Avg):")
    print(avg_metrics)

    avg_df = pd.DataFrame([avg_metrics])
    avg_path = os.path.join(cfg.log.path, f"{cfg.log.dir}_average_metrics.csv")
    avg_df.to_csv(avg_path, index=False)

    print(f"\n? All results saved in: {cfg.log.path}")
    print(f"?? Detailed CSV: {detailed_csv_path}")
    print(f"?? Summary Excel: {summary_path}")
    print(f"?? Average Metrics: {avg_path}")

if __name__ == "__main__":
    main()
