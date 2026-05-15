Vitamin Deficiency Detection via Computer Vision

Overview

Nutrient deficiencies often go undiagnosed until severe physical symptoms manifest. Traditional diagnostic methods (blood panels) are invasive, expensive, and resource-intensive. This project introduces a non-invasive, deep learning-based pre-screening tool designed to analyze visible dermatological and ocular symptoms and detect potential vitamin deficiencies.

Disclaimer: This is a pre-screening triage tool designed to flag high-confidence visual markers and recommend targeted clinical testing. It is not a definitive medical diagnostic tool.

Key Features & Engineering

Training a Convolutional Neural Network (CNN) on a limited, highly imbalanced medical dataset requires advanced optimization. This pipeline implements:

Architectural Upgrades: Utilizes ResNet50 via Transfer Learning to extract deep, complex textural features from skin and eye imagery.

Progressive Unfreezing: Freezes foundational geometric layers while actively training deep convolutional blocks (layer4) to fine-tune specifically for dermatology.

Class Consolidation: Programmatically merges visually identical sub-classes (e.g., various B-vitamins) to reduce mathematical noise and improve predictive reliability.

Weighted Cross-Entropy Loss: Dynamically penalizes the network for misclassifying rare diseases to combat heavy dataset imbalance.

Aggressive Data Augmentation: Implements ColorJitter, RandomRotation, and RandomErasing to prevent overfitting and force the model to learn holistic facial/skin features rather than lighting conditions or background noise.

Smart Training Dynamics & Early Stopping: Utilizes a ReduceLROnPlateau scheduler for precise convergence and Dynamic Early Stopping (patience = 3) to automatically terminate training the moment validation loss degrades, perfectly preserving the healthiest model weights.

The Classes

The model classifies images into one of 8 categories:

General Vitamin B Deficiency (B-Complex, B3, B9, B12)

Mineral and Protein Deficiencies

Vitamin A Deficiency

Vitamin B2 Deficiency

Vitamin C Deficiency

Vitamin D Deficiency

Vitamin E Deficiency

Vitamin K Deficiency

Technology Stack

Framework: PyTorch

Model Architecture: ResNet50 (torchvision.models)

Optimization: Adam Optimizer

Data Processing: Python (os, shutil, zipfile, PIL)

Evaluation & Visualization: Scikit-Learn (classification_report), Matplotlib

Performance Metrics

The model was evaluated using a strict 80/20 Train-Test split.

Overall Test Accuracy: 66.00%

Macro Average F1-Score: 0.74

High-Confidence Detection: Successfully isolated unique visual features for pathognomonic conditions (Vitamins B2, C, E, and K), achieving exceptional F1-scores ranging from 0.96 - 0.99.

Future Scope

Diverse Demographic Training: Expanding the dataset to include a wider variety of Fitzpatrick skin types to mitigate skin-tone bias in the CNN.

Multi-modal Integration: Pairing the image scanner with a digital symptom questionnaire to narrow down "General" classifications (like B-Vitamins) to specific prescriptions.
