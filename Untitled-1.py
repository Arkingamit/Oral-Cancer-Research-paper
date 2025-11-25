# similarly as above sample paper write the same structured and detail for my below code 

# # Enhanced Dataset Class with better error handling
# class ClassifierDataset(Dataset):
#     def __init__(self, annotations_path, transform=None, augment=False):
#         self.annotations_path = annotations_path
#         self.transform = transform
#         self.augment = augment

#         with open(annotations_path, "r") as f:
#             data = json.load(f)

#         self.images_list = data.get("images", [])
#         self.annotations_list = data.get("annotations", [])
#         self.categories_list = data.get("categories", [])

#         self.images = {img["id"]: img for img in self.images_list}
#         self.categories = {cat["id"]: idx for idx, cat in enumerate(self.categories_list)}

#         self.valid_annotations = []
#         for ann in self.annotations_list:
#             bbox = ann.get("bbox", [0, 0, 0, 0])
#             if bbox != [0, 0, 0, 0] and bbox[2] > 0 and bbox[3] > 0:
#                 if ann["image_id"] in self.images:
#                     self.valid_annotations.append(ann)

#         print(f"Dataset loaded: {len(self.valid_annotations)} valid annotations")

#     def __len__(self):
#         return len(self.valid_annotations)

#     def __getitem__(self, idx):
#         ann = self.valid_annotations[idx]
#         img_meta = self.images[ann["image_id"]]

#         img_filename = img_meta.get("file_name")
#         images_dir = "./datasets/oral1/"
#         full_path = os.path.join(images_dir, img_filename)

#         try:
#             image = Image.open(full_path).convert("RGB")
#         except FileNotFoundError:
#             if img_meta.get("path"):
#                 alt_path = "./" + img_meta["path"]
#                 image = Image.open(alt_path).convert("RGB")
#             else:
#                 raise FileNotFoundError(f"Could not find image: {full_path}")

#         x, y, w, h = ann["bbox"]
#         x = max(0, min(x, image.width - 1))
#         y = max(0, min(y, image.height - 1))
#         w = max(1, min(w, image.width - x))
#         h = max(1, min(h, image.height - y))

#         subimage = image.crop((x, y, x + w, y + h))

#         if self.transform:
#             subimage = self.transform(subimage)

#         category_id = ann.get("category_id")
#         label = self.categories.get(category_id)

#         return subimage, label

#     def get_category_names(self):
#         category_names = [None] * len(self.categories_list)
#         for cat in self.categories_list:
#             idx = self.categories[cat["id"]]
#             category_names[idx] = cat["name"]
#         return category_names

# # Enhanced Texture Feature Extractor with more features
# class EnhancedTextureFeatureExtractor:
#     def __init__(self):
#         self.lbp_radii = [1, 2, 3]  # Multiple radii for multi-scale LBP

#     def extract_multiscale_lbp(self, gray_image):
#         """Extract multi-scale LBP features"""
#         features = []
#         for radius in self.lbp_radii:
#             n_points = 8 * radius
#             lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
#             hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
#             hist = hist.astype(float)
#             hist /= (hist.sum() + 1e-7)
#             features.extend(hist)
#         return np.array(features)

#     def extract_enhanced_glcm(self, gray_image):
#         """Extract enhanced GLCM features"""
#         gray_image = (gray_image / 16).astype(np.uint8)  # 16 levels for better granularity

#         distances = [1, 2, 3, 5]
#         angles = [0, 45, 90, 135]
#         features = []

#         for distance in distances:
#             for angle in angles:
#                 glcm = graycomatrix(gray_image, [distance], [np.radians(angle)],
#                                  levels=16, symmetric=True, normed=True)

#                 contrast = graycoprops(glcm, 'contrast')[0, 0]
#                 dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
#                 homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
#                 energy = graycoprops(glcm, 'energy')[0, 0]
#                 correlation = graycoprops(glcm, 'correlation')[0, 0]
#                 asm = graycoprops(glcm, 'ASM')[0, 0]

#                 features.extend([contrast, dissimilarity, homogeneity, energy, correlation, asm])

#         return np.array(features)

#     def extract_gabor_features(self, gray_image):
#         """Extract Gabor filter features with more frequencies"""
#         frequencies = [0.05, 0.1, 0.2, 0.3, 0.4]
#         angles = [0, 30, 60, 90, 120, 150]
#         features = []

#         for frequency in frequencies:
#             for angle in angles:
#                 filt_real, filt_imag = gabor(gray_image, frequency=frequency, theta=np.radians(angle))
#                 features.extend([
#                     filt_real.mean(), filt_real.var(),
#                     filt_imag.mean(), filt_imag.var(),
#                     np.sqrt(filt_real**2 + filt_imag**2).mean()  # Magnitude
#                 ])

#         return np.array(features)

#     def extract_haralick_features(self, gray_image):
#         """Extract extended Haralick features"""
#         try:
#             haralick_features = mahotas.features.haralick(gray_image.astype(np.uint8), compute_14th_feature=True)
#             return np.concatenate([haralick_features.mean(axis=0), haralick_features.std(axis=0)])
#         except:
#             return np.zeros(28)  # 14 mean + 14 std features

#     def extract_statistical_features(self, gray_image):
#         """Extract statistical features"""
#         features = [
#             gray_image.mean(),
#             gray_image.std(),
#             np.median(gray_image),
#             gray_image.min(),
#             gray_image.max(),
#             np.percentile(gray_image, 25),
#             np.percentile(gray_image, 75),
#             measure.shannon_entropy(gray_image)
#         ]
#         return np.array(features)

#     def extract_all_features(self, image_tensor):
#         """Extract all texture features"""
#         if isinstance(image_tensor, torch.Tensor):
#             image_np = image_tensor.permute(1, 2, 0).numpy()
#         else:
#             image_np = np.array(image_tensor)

#         if len(image_np.shape) == 3:
#             gray_image = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#         else:
#             gray_image = (image_np * 255).astype(np.uint8)

#         # Extract all feature types
#         lbp_features = self.extract_multiscale_lbp(gray_image)
#         glcm_features = self.extract_enhanced_glcm(gray_image)
#         gabor_features = self.extract_gabor_features(gray_image)
#         haralick_features = self.extract_haralick_features(gray_image)
#         statistical_features = self.extract_statistical_features(gray_image)

#         all_features = np.concatenate([
#             lbp_features, glcm_features, gabor_features,
#             haralick_features, statistical_features
#         ])

#         return torch.tensor(all_features, dtype=torch.float32)

# # Enhanced Fusion Dataset
# class EnhancedFusionDataset(Dataset):
#     def __init__(self, annotations_path, transform=None):
#         self.base_dataset = ClassifierDataset(annotations_path, transform)
#         self.texture_extractor = EnhancedTextureFeatureExtractor()

#     def __len__(self):
#         return len(self.base_dataset)

#     def __getitem__(self, idx):
#         image, label = self.base_dataset[idx]
#         texture_features = self.texture_extractor.extract_all_features(image)
#         return image, texture_features, label

#     def get_category_names(self):
#         return self.base_dataset.get_category_names()

# # Enhanced CNN Feature Extractor - ONLY EfficientNet-B0
# class EnhancedCNNFeatureExtractor(nn.Module):
#     def __init__(self, output_dim=256):
#         super().__init__()

#         # Using only EfficientNet-B0
#         weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
#         self.backbone = torchvision.models.efficientnet_b0(weights=weights)
#         in_features = self.backbone.classifier[1].in_features
#         self.backbone.classifier = nn.Identity()
#         self.preprocess = weights.transforms()

#         # Feature projection with normalization
#         self.feature_projection = nn.Sequential(
#             nn.Linear(in_features, output_dim),
#             nn.BatchNorm1d(output_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3)
#         )

#     def forward(self, x):
#         x = self.preprocess(x)
#         x = self.backbone(x)
#         x = self.feature_projection(x)
#         return x

# # Advanced Fusion Model with Feature Preservation
# class AdvancedOralCancerFusionModel(LightningModule):
#     def __init__(self, num_classes=3, lr=1e-3,
#                  weight_decay=1e-4, max_epochs=100, texture_feature_dim=500):
#         super().__init__()
#         self.save_hyperparameters()

#         # CNN feature extractor - ONLY EfficientNet-B0
#         self.cnn_dim = 256
#         self.cnn_extractor = EnhancedCNNFeatureExtractor(self.cnn_dim)

#         # Texture feature dimension (dynamically calculated)
#         self.texture_dim = texture_feature_dim

#         # Feature normalization layers to prevent shrinking
#         self.cnn_norm = nn.BatchNorm1d(self.cnn_dim)
#         self.texture_norm = nn.BatchNorm1d(self.texture_dim)

#         # Texture feature enhancement network
#         self.texture_enhancer = nn.Sequential(
#             nn.Linear(self.texture_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU()
#         )

#         # Attention mechanism for feature weighting
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=256, num_heads=8, dropout=0.1, batch_first=True
#         )

#         # Feature fusion with gating mechanism
#         self.gate_cnn = nn.Sequential(
#             nn.Linear(self.cnn_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.cnn_dim),
#             nn.Sigmoid()
#         )

#         self.gate_texture = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.Sigmoid()
#         )

#         # Deep classifier with residual connections
#         self.classifier = nn.Sequential(
#             nn.Linear(self.cnn_dim + 256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )

#         # Loss functions
#         self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

#     def forward(self, image, texture_features):
#         # Extract and normalize CNN features
#         cnn_features = self.cnn_extractor(image)
#         cnn_features = self.cnn_norm(cnn_features)

#         # Normalize and enhance texture features
#         texture_features = texture_features.to(cnn_features.device)
#         texture_features = self.texture_norm(texture_features)
#         enhanced_texture = self.texture_enhancer(texture_features)

#         # Apply cross-attention between CNN and texture features
#         cnn_unsqueeze = cnn_features.unsqueeze(1)
#         texture_unsqueeze = enhanced_texture.unsqueeze(1)

#         attended_cnn, _ = self.cross_attention(
#             cnn_unsqueeze, texture_unsqueeze, texture_unsqueeze
#         )
#         attended_cnn = attended_cnn.squeeze(1)

#         # Apply gating mechanism to preserve important features
#         gated_cnn = cnn_features * self.gate_cnn(cnn_features) + attended_cnn * 0.5
#         gated_texture = enhanced_texture * self.gate_texture(enhanced_texture)

#         # Concatenate features (preserving both)
#         fused_features = torch.cat([gated_cnn, gated_texture], dim=1)

#         # Classification
#         output = self.classifier(fused_features)
#         return output

#     def training_step(self, batch, batch_idx):
#         image, texture_features, label = batch
#         logits = self(image, texture_features)
#         loss = self.loss_fn(logits, label)

#         preds = torch.argmax(logits, dim=1)
#         acc = (preds == label).float().mean()

#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         image, texture_features, label = batch
#         logits = self(image, texture_features)
#         loss = self.loss_fn(logits, label)

#         preds = torch.argmax(logits, dim=1)
#         acc = (preds == label).float().mean()

#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

#         return loss

#     def test_step(self, batch, batch_idx):
#         image, texture_features, label = batch
#         logits = self(image, texture_features)
#         loss = self.loss_fn(logits, label)

#         preds = torch.argmax(logits, dim=1)

#         return {'loss': loss, 'preds': preds, 'labels': label, 'logits': logits}

#     def configure_optimizers(self):
#         # Different learning rates for different parts
#         params = [
#             {'params': self.cnn_extractor.parameters(), 'lr': self.hparams.lr * 0.1},
#             {'params': self.texture_enhancer.parameters(), 'lr': self.hparams.lr},
#             {'params': self.classifier.parameters(), 'lr': self.hparams.lr}
#         ]

#         optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)

#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer,
#             max_lr=self.hparams.lr,
#             total_steps=self.trainer.estimated_stepping_batches,
#             pct_start=0.3,
#             anneal_strategy='cos'
#         )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "step",
#                 "frequency": 1
#             }
#         }

# # Comprehensive Metrics Calculator
# class MetricsCalculator:
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
    
#     def calculate_metrics(self, predictions, targets, probabilities=None):
#         """Calculate comprehensive metrics"""
        
#         # Basic metrics
#         accuracy = accuracy_score(targets, predictions)
#         precision = precision_score(targets, predictions, average='macro', zero_division=0)
#         recall = recall_score(targets, predictions, average='macro', zero_division=0)
#         f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        
#         # Confusion matrix
#         cm = confusion_matrix(targets, predictions)
        
#         # Calculate TP, TN, FP, FN
#         if self.num_classes == 2:
#             if cm.size == 4:
#                 tn, fp, fn, tp = cm.ravel()
#             else:
#                 # Handle case where not all classes are present
#                 tp = tp if 'tp' in locals() else 0
#                 tn = tn if 'tn' in locals() else 0
#                 fp = fp if 'fp' in locals() else 0
#                 fn = fn if 'fn' in locals() else 0
#         else:
#             # Multi-class case
#             tp = np.diag(cm).sum()
#             tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)).sum()
#             fp = cm.sum(axis=0) - np.diag(cm)
#             fn = cm.sum(axis=1) - np.diag(cm)
#             fp = fp.sum()
#             fn = fn.sum()
        
#         # ROC-AUC
#         roc_auc = 0.0
#         if probabilities is not None:
#             try:
#                 if self.num_classes == 2:
#                     if probabilities.shape[1] >= 2:
#                         roc_auc = roc_auc_score(targets, probabilities[:, 1])
#                 else:
#                     # Multi-class ROC-AUC
#                     unique_classes = np.unique(targets)
#                     if len(unique_classes) > 1:
#                         targets_binarized = label_binarize(targets, classes=range(self.num_classes))
#                         # Handle case where some classes might not be present
#                         if targets_binarized.shape[1] == probabilities.shape[1]:
#                             roc_auc = roc_auc_score(targets_binarized, probabilities, 
#                                                   multi_class='ovr', average='macro')
#             except Exception as e:
#                 print(f"Warning: Could not calculate ROC-AUC: {e}")
#                 roc_auc = 0.0
        
#         # Classification report
#         report_dict = classification_report(targets, predictions, output_dict=True, zero_division=0)
        
#         return {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1,
#             'roc_auc': roc_auc,
#             'tp': int(tp),
#             'tn': int(tn),
#             'fp': int(fp),
#             'fn': int(fn),
#             'confusion_matrix': cm,
#             'classification_report': report_dict
#         }

# # Main Training Function - 10 Iterations with EfficientNet-B0
# def run_efficientnet_10_iterations(batch_size=32, max_epochs=50):
#     """Run EfficientNet-B0 model for 10 iterations and generate comprehensive results"""
    
#     print("="*80)
#     print("ORAL CANCER DETECTION - EFFICIENTNET-B0 - 10 ITERATIONS")
#     print("="*80)
    
#     # Configuration
#     NUM_RUNS = 10
#     model_name = "efficientnet_b0"
    
#     # Create results directory
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = f"./results/efficientnet_b0_10runs_{timestamp}"
#     os.makedirs(results_dir, exist_ok=True)
    
#     # Store all results
#     all_detailed_results = []
#     all_summary_results = []
    
#     # Store timing information
#     total_experiment_start = time.time()
    
#     for run_idx in range(NUM_RUNS):
#         print(f"\n{'='*60}")
#         print(f"RUN {run_idx + 1}/{NUM_RUNS} - EFFICIENTNET-B0")
#         print(f"{'='*60}")
        
#         # Set different seed for each run for variability
#         run_seed = SEED + run_idx * 1000
#         seed_everything(run_seed, workers=True)
#         torch.manual_seed(run_seed)
#         np.random.seed(run_seed)
        
#         print(f"Run Seed: {run_seed}")
        
#         # Paths
#         train_path = "./datasets/train.json"
#         val_path = "./datasets/val.json"
#         test_path = "./datasets/test.json"

#         # Enhanced transforms with more augmentation
#         train_transform = T.Compose([
#             T.Resize((256, 256), antialias=True),
#             T.RandomCrop(224),
#             T.RandomHorizontalFlip(p=0.5),
#             T.RandomVerticalFlip(p=0.3),
#             T.RandomRotation(20),
#             T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
#             T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#             T.RandomPerspective(distortion_scale=0.2, p=0.3),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             T.RandomErasing(p=0.2)
#         ])

#         val_transform = T.Compose([
#             T.Resize((256, 256), antialias=True),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         # Create datasets
#         train_dataset = EnhancedFusionDataset(train_path, train_transform)
#         val_dataset = EnhancedFusionDataset(val_path, val_transform)
#         test_dataset = EnhancedFusionDataset(test_path, val_transform)

#         # Calculate texture feature dimension
#         sample_img, sample_texture, _ = train_dataset[0]
#         texture_dim = sample_texture.shape[0]
#         num_classes = len(train_dataset.get_category_names())

#         if run_idx == 0:  # Print dataset info only once
#             print(f"\nDataset Statistics:")
#             print(f"  Train: {len(train_dataset)} samples")
#             print(f"  Val: {len(val_dataset)} samples")
#             print(f"  Test: {len(test_dataset)} samples")
#             print(f"  Categories: {train_dataset.get_category_names()}")
#             print(f"  Texture features dimension: {texture_dim}")

#         # Data loaders
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=4,
#             pin_memory=True,
#             persistent_workers=True
#         )

#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             persistent_workers=True
#         )

#         test_loader = DataLoader(
#             test_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4,
#             pin_memory=True,
#             persistent_workers=True
#         )

#         # Initialize model
#         model = AdvancedOralCancerFusionModel(
#             num_classes=num_classes,
#             lr=1e-3,
#             weight_decay=1e-4,
#             max_epochs=max_epochs,
#             texture_feature_dim=texture_dim
#         )

#         # Create run directory
#         run_dir = os.path.join(results_dir, f"run_{run_idx + 1}")
#         os.makedirs(run_dir, exist_ok=True)

#         # Callbacks
#         checkpoint_callback = ModelCheckpoint(
#             monitor="val_loss",
#             mode="min",
#             save_top_k=1,
#             filename="best-checkpoint",
#             save_last=True,
#             dirpath=run_dir
#         )

#         early_stopping = EarlyStopping(
#             monitor="val_loss",
#             mode="min",
#             patience=15,
#             verbose=True,
#             min_delta=0.001
#         )

#         lr_monitor = LearningRateMonitor(logging_interval='epoch')

#         # Trainer
#         trainer = Trainer(
#             max_epochs=max_epochs,
#             callbacks=[checkpoint_callback, early_stopping, lr_monitor],
#             accelerator="gpu" if torch.cuda.is_available() else "cpu",
#             devices=1,
#             precision="16-mixed" if torch.cuda.is_available() else 32,
#             gradient_clip_val=1.0,
#             accumulate_grad_batches=2,
#             deterministic=True,
#             enable_progress_bar=True,
#             log_every_n_steps=10,
#             enable_model_summary=False  # Reduce output verbosity
#         )

#         # Training
#         print(f"Starting training for Run {run_idx + 1}...")
#         train_start_time = time.time()
#         trainer.fit(model, train_loader, val_loader)
#         train_end_time = time.time()
#         train_time = train_end_time - train_start_time

#         # Initialize metrics calculator
#         metrics_calc = MetricsCalculator(num_classes)
#         device = next(model.parameters()).device

#         # ================ TRAINING METRICS CALCULATION ================ #
#         print(f"Evaluating training data for Run {run_idx + 1}...")
#         model.eval()
        
#         train_preds, train_targets, train_probs = [], [], []
#         with torch.no_grad():
#             for batch in train_loader:
#                 images, texture_features, labels = batch
#                 images = images.to(device)
#                 texture_features = texture_features.to(device)
#                 labels = labels.to(device)
                
#                 logits = model(images, texture_features)
#                 probs = F.softmax(logits, dim=1)
#                 preds = torch.argmax(probs, dim=1)
                
#                 train_preds.append(preds.cpu())
#                 train_targets.append(labels.cpu())
#                 train_probs.append(probs.cpu())

#         train_preds = torch.cat(train_preds).numpy()
#         train_targets = torch.cat(train_targets).numpy()
#         train_probs = torch.cat(train_probs).numpy()

#         # Calculate training metrics
#         train_metrics = metrics_calc.calculate_metrics(
#             train_preds, train_targets, train_probs
#         )

#         # ================ TESTING PHASE ================ #
#         print(f"Evaluating test data for Run {run_idx + 1}...")
#         test_start_time = time.time()
        
#         # Load best checkpoint
#         best_model_path = checkpoint_callback.best_model_path
#         if best_model_path:
#             model = AdvancedOralCancerFusionModel.load_from_checkpoint(
#                 best_model_path,
#                 num_classes=num_classes,
#                 texture_feature_dim=texture_dim
#             )
        
#         model.eval()
#         model = model.to(device)
        
#         test_preds, test_targets, test_probs = [], [], []
#         with torch.no_grad():
#             for batch in test_loader:
#                 images, texture_features, labels = batch
#                 images = images.to(device)
#                 texture_features = texture_features.to(device)
#                 labels = labels.to(device)
                
#                 logits = model(images, texture_features)
#                 probs = F.softmax(logits, dim=1)
#                 preds = torch.argmax(probs, dim=1)
                
#                 test_preds.append(preds.cpu())
#                 test_targets.append(labels.cpu())
#                 test_probs.append(probs.cpu())
        
#         test_end_time = time.time()
#         test_time = test_end_time - test_start_time

#         test_preds = torch.cat(test_preds).numpy()
#         test_targets = torch.cat(test_targets).numpy()
#         test_probs = torch.cat(test_probs).numpy()

#         # Calculate test metrics
#         test_metrics = metrics_calc.calculate_metrics(
#             test_preds, test_targets, test_probs
#         )

#         # Save individual run results
#         # Training detailed results
#         train_result = {
#             'Iteration': f'Run_{run_idx + 1}',
#             'Seed': run_seed,
#             'Model': 'efficientnet_b0',
#             'Phase': 'Train',
#             'Accuracy': train_metrics['accuracy'],
#             'Precision': train_metrics['precision'],
#             'Recall': train_metrics['recall'],
#             'F1-score': train_metrics['f1_score'],
#             'ROC_AUC': train_metrics['roc_auc'],
#             'TP': train_metrics['tp'],
#             'TN': train_metrics['tn'],
#             'FP': train_metrics['fp'],
#             'FN': train_metrics['fn'],
#             'Time_sec': round(train_time, 2)
#         }
#         all_detailed_results.append(train_result)

#         # Test detailed results
#         test_result = {
#             'Iteration': f'Run_{run_idx + 1}',
#             'Seed': run_seed,
#             'Model': 'efficientnet_b0',
#             'Phase': 'Test',
#             'Accuracy': test_metrics['accuracy'],
#             'Precision': test_metrics['precision'],
#             'Recall': test_metrics['recall'],
#             'F1-score': test_metrics['f1_score'],
#             'ROC_AUC': test_metrics['roc_auc'],
#             'TP': test_metrics['tp'],
#             'TN': test_metrics['tn'],
#             'FP': test_metrics['fp'],
#             'FN': test_metrics['fn'],
#             'Time_sec': round(test_time, 2)
#         }
#         all_detailed_results.append(test_result)

#         # Summary for this run
#         run_summary = {
#             'run': run_idx + 1,
#             'seed': run_seed,
#             'model': 'efficientnet_b0',
#             'train_time': round(train_time, 2),
#             'test_time': round(test_time, 2),
#             'total_time': round(train_time + test_time, 2),
#             'train_accuracy': round(train_metrics['accuracy'], 4),
#             'train_precision': round(train_metrics['precision'], 4),
#             'train_recall': round(train_metrics['recall'], 4),
#             'train_f1_score': round(train_metrics['f1_score'], 4),
#             'train_roc_auc': round(train_metrics['roc_auc'], 4),
#             'test_accuracy': round(test_metrics['accuracy'], 4),
#             'test_precision': round(test_metrics['precision'], 4),
#             'test_recall': round(test_metrics['recall'], 4),
#             'test_f1_score': round(test_metrics['f1_score'], 4),
#             'test_roc_auc': round(test_metrics['roc_auc'], 4)
#         }
#         all_summary_results.append(run_summary)

#         # Save individual run reports
#         train_report_df = pd.DataFrame(train_metrics['classification_report']).transpose()
#         train_report_df.to_excel(os.path.join(run_dir, f"classification_report_train_run{run_idx + 1}.xlsx"))
        
#         train_cm_df = pd.DataFrame(train_metrics['confusion_matrix'])
#         train_cm_df.to_excel(os.path.join(run_dir, f"confusion_matrix_train_run{run_idx + 1}.xlsx"))
        
#         test_report_df = pd.DataFrame(test_metrics['classification_report']).transpose()
#         test_report_df.to_excel(os.path.join(run_dir, f"classification_report_test_run{run_idx + 1}.xlsx"))
        
#         test_cm_df = pd.DataFrame(test_metrics['confusion_matrix'])
#         test_cm_df.to_excel(os.path.join(run_dir, f"confusion_matrix_test_run{run_idx + 1}.xlsx"))

#         # Print run summary
#         print(f"\n{'='*50}")
#         print(f"RUN {run_idx + 1} COMPLETED")
#         print(f"{'='*50}")
#         print(f"Seed: {run_seed}")
#         print(f"Train Time: {train_time:.2f}s | Test Time: {test_time:.2f}s")
#         print(f"\nTRAIN METRICS:")
#         print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
#         print(f"  Precision: {train_metrics['precision']:.4f}")
#         print(f"  Recall: {train_metrics['recall']:.4f}")
#         print(f"  F1-Score: {train_metrics['f1_score']:.4f}")
#         print(f"  ROC-AUC: {train_metrics['roc_auc']:.4f}")
#         print(f"\nTEST METRICS:")
#         print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
#         print(f"  Precision: {test_metrics['precision']:.4f}")
#         print(f"  Recall: {test_metrics['recall']:.4f}")
#         print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
#         print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")

#         # Save intermediate results every 2 runs
#         if (run_idx + 1) % 2 == 0:
#             intermediate_df = pd.DataFrame(all_detailed_results)
#             intermediate_path = os.path.join(results_dir, f"intermediate_results_run{run_idx + 1}.csv")
#             intermediate_df.to_csv(intermediate_path, index=False)
#             print(f"Intermediate results saved: {intermediate_path}")

#     # Calculate total experiment time
#     total_experiment_end = time.time()
#     total_experiment_time = total_experiment_end - total_experiment_start

#     print(f"\n{'='*80}")
#     print(f"ALL 10 RUNS COMPLETED!")
#     print(f"Total Experiment Time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")
#     print(f"{'='*80}")

#     # ================ SAVE ALL FINAL RESULTS ================ #
    
#     # Convert to DataFrames
#     detailed_df = pd.DataFrame(all_detailed_results)
#     summary_df = pd.DataFrame(all_summary_results)
    
#     # Save detailed results (main CSV file)
#     detailed_csv_path = os.path.join(results_dir, "efficientnet_b0_10runs_detailed_results.csv")
#     detailed_df.to_csv(detailed_csv_path, index=False)
    
#     # Save summary results
#     summary_excel_path = os.path.join(results_dir, "efficientnet_b0_10runs_summary.xlsx")
#     summary_df.to_excel(summary_excel_path, index=False)
    
#     # Calculate statistics across all runs
#     test_results = summary_df.copy()
  
# # Main execution function
# if __name__ == "__main__":
#     print("="*80)
#     print("ORAL CANCER DETECTION - EFFICIENTNET-B0 - 10 ITERATIONS")
#     print(f"Base Seed: {SEED}")
#     print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
#     print("="*80)

#     # Configuration
#     BATCH_SIZE = 32
#     MAX_EPOCHS = 50

#     print(f"\nConfiguration:")
#     print(f"  Model: EfficientNet-B0")
#     print(f"  Batch Size: {BATCH_SIZE}")
#     print(f"  Max Epochs: {MAX_EPOCHS}")
#     print(f"  Number of Runs: 10")
#     print("="*80)

#     # Run 10 iterations
#     results = run_efficientnet_10_iterations(
#         batch_size=BATCH_SIZE,
#         max_epochs=MAX_EPOCHS
#     )
    
#     # Create comprehensive visualizations
#     if results:
#         create_visualization_10_runs(results)
    
#     print(f"\n{'='*80}")
#     print("10 ITERATIONS COMPLETED SUCCESSFULLY!")
#     print("="*80)
#     print(f"Main CSV file location: {results['main_csv_path']}")
#     print(f"Results directory: {results['results_directory']}")
#     print("="*80)


values = [
0.8004926108374385,
0.8572967980295566,
0.7756055008210181,
0.8260467980295566,
0.8200944170771757,
0.8260467980295566,
0.795617816091954,
0.7777606732348111,
0.8115763546798029,
0.8011596880131363
]

# Calculate the average
average = sum(values) / len(values)
print(average)
