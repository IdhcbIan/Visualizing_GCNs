# Visualizing Graph Convolutional Networks (GCNs)

A comprehensive framework for image classification using Graph Convolutional Networks with multiple feature extractors and graph construction methods. This project combines deep learning feature extraction with graph-based learning for enhanced image classification performance.

## 🚀 Overview

This repository implements a complete pipeline that:
1. **Extracts features** from images using 50+ pre-trained deep learning models
2. **Constructs graphs** from image features using k-nearest neighbors or reciprocal neighbors
3. **Trains GCN models** for image classification using various graph neural network architectures
4. **Visualizes results** including embeddings, graphs, and classification performance

## ✨ Features

### Feature Extractors (50+ Models)
- **Vision Transformers**: ViT, DINOv2 (ViTS14, ViTB14, ViTL14, ViTG14)
- **ResNet Family**: ResNet18/34/50/101/152, SE-ResNet variants
- **ConvNeXt**: ConvNeXt-Tiny, ConvNeXt-Large
- **Swin Transformer**: SwinT
- **DenseNet**: DenseNet121/161/169/201
- **VGG**: VGG11/13/16/19 (with and without BatchNorm)
- **Inception**: InceptionV4, InceptionResNetV2
- **And many more**: Xception, SENet, PNASNet, PolyNet, etc.

### Graph Neural Networks (5 Architectures)
- **GCN**: Standard Graph Convolutional Network
- **GAT**: Graph Attention Network
- **SGC**: Simplified Graph Convolution
- **APPNP**: Approximate Personalized Propagation of Neural Predictions
- **ARMA**: Autoregressive Moving Average Graph Filter

### Graph Construction Methods
- **K-Nearest Neighbors (KNN)**: Connect each node to its k nearest neighbors

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Key Dependencies
- PyTorch & torchvision
- PyTorch Geometric
- scikit-learn
- NumPy, Matplotlib, Seaborn
- Transformers, timm
- NetworkX for graph visualization
- UMAP for dimensionality reduction

## 🚀 Quick Start

### 1. Feature Extraction

```bash
cd Extractor_Inference_Code
python Main.py
```

This will:
1. Prompt you to select a feature extractor model
2. Process all images in the `imgs/` directory
3. Generate feature embeddings and save them as `.npy` files
4. Create ranking files for graph construction

### 2. GCN Training and Inference

```bash
cd ../GCN_Inference_Code
python Main.py
```

This will:
1. Prompt you to select a GCN model
2. Load the corresponding feature embeddings
3. Construct graphs using k-nearest neighbors
4. Train the GCN model with cross-validation
5. Report classification accuracy


## 📁 Project Structure

```
Visualizing_GCNs/
├── Extractor_Inference_Code/          # Feature extraction pipeline
│   ├── Main.py                        # Main extraction script
│   ├── ViT.py                         # Vision Transformer implementation
│   ├── DinoV2.py                      # DINOv2 implementation
│   ├── ConvNext.py                    # ConvNeXt implementation
│   ├── SwinT.py                       # Swin Transformer implementation
│   ├── Graph.py                       # Graph construction and visualization
│   ├── PlotEmb.py                     # Embedding visualization
│   ├── imgs/                          # Input images directory
│   ├── Emb/                           # Generated embeddings
│   └── Runs/                          # Ranking files
│
├── GCN_Inference_Code/                # GCN training and inference
│   ├── Main.py                        # Main GCN script
│   ├── Lib.py                         # Model options library
│   ├── Tools.py                       # Training utilities
│   ├── utils.py                       # Helper functions and visualization
│   └── Classes.txt                    # Class labels mapping
│
├── Graph_System/                      # Additional graph utilities
├── References/                        # Reference materials and papers
└── Report/                           # Project documentation and reports
```

## 🔧 Configuration

### Key Parameters

- **Feature Extractor**: Choose from 50+ available models
- **GCN Architecture**: Select from 5 different GNN variants
- **Graph Construction**: KNN or reciprocal nearest neighbors
- **Number of Neighbors (k)**: Default 10, adjustable
- **Hidden Neurons**: Default 32, configurable
- **Cross-validation Folds**: Default 10
- **Number of Executions**: Default 5 for statistical robustness

**Author**: Ian Bezerra - ICMC-USP  
**Year**: 2025 