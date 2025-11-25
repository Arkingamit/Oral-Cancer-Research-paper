import hydra
import os
import json
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datasets.classifier_dataset import ClassifierDataset

# class ClassificationDataModule(LightningDataModule):
#     def __init__(self,
#                  train: str,
#                  val: str,
#                  test: str,
#                  batch_size: int = 32,
#                  num_workers: int = 4,
#                  train_transform=None,
#                  val_transform=None,
#                  test_transform=None,
#                  transform=None):
#         super().__init__()
#         # Fallback to common transform if specific not provided
#         if train_transform is None:
#             train_transform = transform
#         if val_transform is None:
#             val_transform = transform
#         if test_transform is None:
#             test_transform = transform

#         # Initialize datasets using the custom Dataset class
#         self.train_dataset = ClassifierDataset(train, transform=train_transform)
#         self.val_dataset   = ClassifierDataset(val,   transform=val_transform)
#         self.test_dataset  = ClassifierDataset(test,  transform=test_transform)

#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers
#         )

#     def teardown(self, stage=None):
#         # Optional cleanup
#         pass
class ClassificationDataModule(LightningDataModule):
    def __init__(self,
                 train: str,
                 val: str,
                 test: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_transform=None,
                 val_transform=None,
                 test_transform=None,
                 transform=None):
        super().__init__()
        if train_transform is None:
            train_transform = transform
        if val_transform is None:
            val_transform = transform
        if test_transform is None:
            test_transform = transform

        self.train_dataset = ClassifierDataset(train, transform=train_transform)
        self.val_dataset   = ClassifierDataset(val,   transform=val_transform)
        self.test_dataset  = ClassifierDataset(test,  transform=test_transform)

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def teardown(self, stage=None):
        pass
