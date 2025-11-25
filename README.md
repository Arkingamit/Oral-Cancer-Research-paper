# Oral Cancer Research Paper

A comprehensive deep learning research project for oral cancer detection and classification using CNN-based architectures and multi-modal feature fusion approaches.

## Overview

This project implements multiple neural network architectures to detect and classify oral cancer from medical images. It combines CNN-only approaches with advanced fusion techniques and texture analysis methods to achieve robust classification performance.

**Repository**: [Oral-Cancer-Research-paper](https://github.com/Arkingamit/Oral-Cancer-Research-paper)

## Project Structure

```
.
├── py files/                          # Main Python scripts and utilities
│   ├── classifier-train.py           # Main CNN classification training script
│   ├── triplet-train.py              # Triplet loss-based training
│   ├── rank-projection.py            # Ranking and projection utilities
│   ├── segregation.py                # Dataset segregation utilities
│   ├── utils.py                      # Common utilities and helpers
│   ├── log.py                        # Logging utilities
│   ├── metrics.py                    # Metrics calculation
│   ├── models/                       # Model architectures
│   ├── datasets/                     # Dataset handling modules
│   ├── scripts/                      # Utility scripts
│   └── requirements.txt              # Python dependencies
│
├── checkpoints/                       # Saved model checkpoints
│   ├── efficientnet_b0/
│   ├── resnet50/
│   └── convnext_tiny/
│
├── cnn only/                          # CNN-only approach results
│   ├── oral_average_metrics.csv
│   ├── oral_detailed_results.csv
│   └── oral_run1-10/
│
├── Fusion/                            # Multi-modal fusion approach results
│   ├── efficientnet_b0_10runs_detailed_results.csv
│   ├── efficientnet_b0_10runs_statistics.csv
│   ├── final_experiment_report.json
│   └── run_1-10/
│
├── texture_only/                      # Texture-only analysis results
│   ├── texture_only_10runs_detailed_results.csv
│   ├── texture_only_10runs_statistics.csv
│   ├── final_experiment_report.json
│   └── run_1-10/
│
├── results/                           # Comprehensive experiment results
│   └── model_comparison_comprehensive.csv
│
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## Key Features

### 🔬 Multiple Approaches

1. **CNN-Only Classification**
   - Pure CNN-based image classification
   - Baseline for comparison

2. **Multi-Modal Fusion**
   - Combines CNN features with texture features
   - Advanced fusion techniques for improved accuracy
   - Results: ~80.67% average accuracy

3. **Texture Analysis**
   - Texture-based feature extraction
   - Analysis of lesion characteristics

### 🏗️ Supported Architectures

- **EfficientNet-B0**: Lightweight and efficient
- **ResNet50**: Deep residual network
- **ConvNeXt Tiny**: Modern vision transformer-based architecture

### 📊 Experimental Setup

- **10 runs per configuration** for statistical robustness
- **Cross-validation** and rigorous evaluation
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Random seed management** for reproducibility

## Installation

### Requirements

- Python 3.8+
- CUDA/GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Arkingamit/Oral-Cancer-Research-paper.git
cd Oral-Cancer-Research-paper
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r py\ files/requirements.txt
```

## Key Dependencies

- **PyTorch Lightning** (2.0.8): Deep learning framework
- **PyTorch**: Neural network library
- **TensorFlow** (2.13.0): Alternative deep learning library
- **Scikit-learn** (1.3.0): Machine learning utilities
- **Pandas** (2.1.0): Data manipulation
- **Numpy** (1.24.1): Numerical computing
- **Hydra** (1.3.2): Configuration management
- **WandB** (0.15.10): Experiment tracking
- **Pillow** (9.3.0): Image processing

## Usage

### Training a CNN Classifier

```bash
python py\ files/classifier-train.py
```

Supports configuration via Hydra config files located in `./config/`

### Training with Triplet Loss

```bash
python py\ files/triplet-train.py
```

### Running Rank Projection

```bash
python py\ files/rank-projection.py
```

### Dataset Segregation

```bash
python py\ files/segregation.py
```

## Experimental Results

### CNN-Only Approach
- Average test accuracy across 10 runs
- Located in: `cnn only/oral_average_metrics.csv`

### Fusion Approach (EfficientNet-B0)
- **Average Accuracy**: 80.67% (±2.35%)
- **Average Precision**: 81.2% (±2.21%)
- **Average Recall**: 80.92% (±2.46%)
- **Average F1-Score**: 80.8% (±2.28%)
- **Average ROC-AUC**: 93.87% (±1.59%)
- **Best Accuracy**: 85.39% (Run 2)

Results file: `Fusion/final_experiment_report.json`

### Texture-Only Approach
- Detailed results in: `texture_only/final_experiment_report.json`

## Model Checkpoints

Pre-trained model checkpoints are available in the `checkpoints/` directory:

- **EfficientNet-B0**: Multiple epoch checkpoints and final model
- **ResNet50**: Training checkpoints available
- **ConvNeXt Tiny**: Checkpoint structure ready

### Loading Checkpoints

```python
import torch
checkpoint = torch.load('checkpoints/efficientnet_b0/efficientnet_b0_final.ckpt')
# Load into your model
```

## Configuration

Configuration is managed using Hydra. Config files are located in `./config/`:

- `config_classification.yaml`: Classification training config
- `config_triplet.yaml`: Triplet loss training config
- `config_fusion.yaml`: Fusion approach config

Example configuration parameters:
- `model.name`: Architecture selection (efficientnet_b0, resnet50, convnext_tiny)
- `model.weights`: Pretrained weights (imagenet, etc.)
- `train.lr`: Learning rate
- `train.max_epochs`: Maximum epochs
- `dataset.train`: Training dataset path
- `dataset.val`: Validation dataset path
- `dataset.test`: Test dataset path

## Data Organization

The project expects datasets in the following format:

- `train.json`: Training dataset manifest
- `val.json`: Validation dataset manifest
- `test.json`: Test dataset manifest
- `dataset.json`: Complete dataset information
- `coco_dataset.json`: COCO-formatted dataset

## Scripts & Utilities

### Core Scripts
- `classifier-train.py`: Main classification training script
- `triplet-train.py`: Metric learning with triplet loss
- `rank-projection.py`: Ranking and projection analysis
- `segregation.py`: Dataset segregation utilities
- `utils.py`: Common utility functions

### Logging & Metrics
- `log.py`: Logging configuration
- `loss_log.py`: Loss tracking utilities
- `metrics.py`: Evaluation metrics

## Results & Analysis

### Metrics Files
- `experiment_metrics.csv`: Individual experiment results
- `oral_ranked_dataset.csv`: Dataset rankings
- `ranking.csv`: Performance rankings
- `model_comparison_comprehensive.csv`: Cross-model comparison

## Citation

If you use this project in your research, please cite the associated paper:

```bibtex
@article{OralCancerResearch2024,
  title={[Your Paper Title]},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Arkin

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Thanks to the deep learning and medical imaging communities
- PyTorch Lightning for the excellent training framework
- Hydra for flexible configuration management
- All contributors and researchers involved in this project

## Contact

For questions or inquiries about this project, please open an issue on GitHub or contact the maintainers.

## Repository

- **GitHub**: [Arkingamit/Oral-Cancer-Research-paper](https://github.com/Arkingamit/Oral-Cancer-Research-paper)
- **Branch**: main

---

**Last Updated**: November 25, 2025
