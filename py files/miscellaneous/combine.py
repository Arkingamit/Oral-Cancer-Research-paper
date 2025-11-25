import os
import time
import numpy as np
import pandas as pd
import cv2
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)

from src.datasets.classifier_datamodule import ClassificationDataModule
from src.models.c import OralClassifierModule
import torchvision.transforms as T

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

###############################
# TEXTURE FEATURE ON CROP
###############################
def extract_texture_features_from_tensor(img_tensor):
    """Compute GLCM + LBP features from an image tensor."""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy() * 255.0
    img_np = img_np.astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

    glcm = graycomatrix(gray, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    glcm_feats = [graycoprops(glcm, p).mean() for p in props]

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    return np.concatenate([glcm_feats, lbp_hist])

###############################
# EXTRACT DEEP+TEXTURE FEATURES
###############################
def extract_hybrid_features(model, dataloader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            _, deep = model(imgs)                        # deep: (B, D)
            deep = deep.cpu().numpy()

            tex = np.stack([
                extract_texture_features_from_tensor(img)
                for img in imgs.cpu()
            ], axis=0)                                  # tex: (B, T)

            feats.append(np.concatenate([deep, tex], 1))
            labels.extend(lbls.numpy())
    return np.vstack(feats), np.array(labels)

###############################
# EVALUATE
###############################
def evaluate(y, y_pred, y_prob, n_classes):
    cm = confusion_matrix(y, y_pred)
    rpt = classification_report(y, y_pred, output_dict=True)
    if n_classes == 2:
        auc = roc_auc_score(y, y_prob[:,1])
    else:
        yb = label_binarize(y, classes=np.arange(n_classes))
        auc = roc_auc_score(yb, y_prob, multi_class='ovr', average='macro')

    # compute TP/TN/FP/FN
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.diag(cm).sum()
        tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)).sum()
        fp = (cm.sum(axis=0) - np.diag(cm)).sum()
        fn = (cm.sum(axis=1) - np.diag(cm)).sum()

    return {
        'Precision': rpt['macro avg']['precision'],
        'Recall': rpt['macro avg']['recall'],
        'F1-score': rpt['macro avg']['f1-score'],
        'Accuracy': rpt['accuracy'],
        'ROC_AUC': auc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    }

###############################
# MAIN PIPELINE (5 runs)
###############################
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    transform = T.Compose([T.Resize((224,224)), T.ToTensor()])
    dm = ClassificationDataModule(
        train='datasets/train.json',
        val  ='datasets/val.json',
        test ='datasets/test.json',
        batch_size=8, num_workers=4,
        transform=transform
    )
    train_loader, test_loader = dm.train_dataloader(), dm.test_dataloader()
    n_classes = len(dm.train_dataset.categories)

    records = []
    for run in range(1,6):
        seed = int.from_bytes(os.urandom(4), 'big')
        torch.manual_seed(seed); np.random.seed(seed)

        # Feature extraction
        model = OralClassifierModule(
            model='resnet50', weights='IMAGENET1K_V1',
            num_classes=n_classes, lr=1e-4,
            max_epochs=1, features_size=2048, frozen_layers=0
        ).to(device)

        t0 = time.time()
        X_train, y_train = extract_hybrid_features(model, train_loader, device)
        train_time = time.time() - t0

        t1 = time.time()
        X_test,  y_test  = extract_hybrid_features(model, test_loader, device)
        test_time = time.time() - t1

        # Standardize
        scaler = StandardScaler().fit(X_train)
        Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

        # Random Forest
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1
        )
        clf.fit(Xtr, y_train)

        # Eval train
        y_tr_pred = clf.predict(Xtr)
        y_tr_prob = clf.predict_proba(Xtr)
        m_tr = evaluate(y_train, y_tr_pred, y_tr_prob, n_classes)
        m_tr.update({'Run':run,'Phase':'Train','Seed':seed,'Time_sec':round(train_time,2)})
        records.append(m_tr)

        # Eval test
        y_te_pred = clf.predict(Xte)
        y_te_prob = clf.predict_proba(Xte)
        m_te = evaluate(y_test, y_te_pred, y_te_prob, n_classes)
        m_te.update({'Run':run,'Phase':'Test','Seed':seed,'Time_sec':round(test_time,2)})
        records.append(m_te)

        print(f"Run {run}: Train Acc {m_tr['Accuracy']:.3f}, Test Acc {m_te['Accuracy']:.3f}")

    df = pd.DataFrame(records)
    os.makedirs('results',exist_ok=True)
    df.to_csv('results/hybrid_RF_5runs.csv', index=False)
    print("\nFinal average:\n", df.groupby('Phase').mean(numeric_only=True))

if __name__=='__main__':
    main()
