# Vitamin Deficiency Detection via Computer Vision

## Overview
Nutrient deficiencies often go undiagnosed until severe physical symptoms manifest. Traditional diagnostic methods (blood panels) are invasive, expensive, and resource-intensive. This project introduces a **non-invasive, deep learning-based pre-screening tool** designed to analyze visible dermatological and ocular symptoms and detect potential vitamin deficiencies. 

*Disclaimer: This is a pre-screening triage tool designed to flag high-confidence visual markers and recommend targeted clinical testing. It is not a definitive medical diagnostic tool.*

## Key Features & Engineering
Training a Convolutional Neural Network (CNN) on a limited, highly imbalanced medical dataset requires advanced optimization. This pipeline implements:
* **Architectural Upgrades:** Utilizes **ResNet50** via Transfer Learning to extract deep, complex textural features from skin and eye imagery.
* **Progressive Unfreezing:** Freezes foundational geometric layers while actively training deep convolutional blocks (`layer4`) to fine-tune specifically for dermatology.
* **Class Consolidation:** Programmatically merges visually identical sub-classes (e.g., various B-vitamins) to reduce mathematical noise and improve predictive reliability.
* **Weighted Cross-Entropy Loss:** Dynamically penalizes the network for misclassifying rare diseases to combat heavy dataset imbalance.
* **Aggressive Data Augmentation:** Implements `ColorJitter`, `RandomRotation`, and `RandomErasing` to prevent overfitting and force the model to learn holistic facial/skin features rather than lighting conditions or background noise.
* **Smart Training Dynamics:** Utilizes a `ReduceLROnPlateau` scheduler for precise convergence and automated model checkpointing to save the optimal weight state.

## The Classes
The model classifies images into one of 8 categories:
1. General Vitamin B Deficiency (B-Complex, B3, B9, B12)
2. Mineral and Protein Deficiencies
3. Vitamin A Deficiency
4. Vitamin B2 Deficiency
5. Vitamin C Deficiency
6. Vitamin D Deficiency
7. Vitamin E Deficiency
8. Vitamin K Deficiency

## Technology Stack
* **Framework:** PyTorch
* **Model Architecture:** ResNet50 (`torchvision.models`)
* **Optimization:** Adam Optimizer
* **Data Processing:** Python (`os`, `shutil`, `zipfile`, `PIL`)
* **Evaluation & Visualization:** Scikit-Learn (`classification_report`), Matplotlib

## Performance Metrics
The model was evaluated using a strict 80/20 Train-Test split. 
* **Overall Test Accuracy:** 65.12%
* **Macro Average F1-Score:** 0.73
* **High-Confidence Detection:** Successfully isolated unique visual features for Vitamins B2, C, E, and K, achieving F1-scores of **0.94 - 0.99**.

## Repository Structure
* `master_script.py` : The complete end-to-end pipeline (Data Extraction -> Augmentation -> Training -> Evaluation).
* `training_metrics_graph.png` : Dual-axis graph visualizing Model Loss and Accuracy over epochs.
* `best_vitamin_model.pth` : The saved weights of the best-performing model checkpoint. 

## Future Scope
* **Diverse Demographic Training:** Expanding the dataset to include a wider variety of Fitzpatrick skin types to mitigate skin-tone bias in the CNN.
* **Multi-modal Integration:** Pairing the image scanner with a digital symptom questionnaire to narrow down "General" classifications (like B-Vitamins) to specific prescriptions.
