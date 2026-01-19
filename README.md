Cost-efficient Disease Diagnosis using Active and Transfer Learning

Overview

This project implements a cost-efficient disease diagnosis framework using
transfer learning and active learning techniques applied to chest X-ray images.
The aim is to reduce labeling effort while maintaining strong diagnostic
performance for multi-label disease classification.

Key Features

* Transfer learning using a ResNet50 model pretrained on ImageNet
* Multi-label classification of chest X-ray diseases
* Active learning with entropy-based uncertainty sampling
* Focal loss and class-weighting to handle class imbalance
* Validation-based threshold tuning for improved F1-score
* Evaluation using Macro F1-score and ROC-AUC

Methodology

The model uses a pretrained ResNet50 backbone with a custom classification head
for multi-label prediction. An active learning loop is applied where the model
iteratively selects the most informative unlabeled samples based on prediction
uncertainty. These samples are then added to the labeled pool to improve model
performance efficiently.

Dataset

The project is designed for chest X-ray datasets containing multiple disease
labels per image. Metadata is provided via a CSV file, and corresponding images
are loaded from a directory structure. Only valid image paths are used during
training and evaluation.

Evaluation Metrics

* Macro F1-score
* Per-class F1-score
* Area Under the ROC Curve (AUC)

Usage

Before running the code, update the dataset paths in the main script:


Then execute the main training script to start the active learning pipeline.

Results

Model performance is evaluated on a held-out test set after completing all
active learning rounds. Training loss, validation F1-score, and AUC are tracked
and saved for analysis.

Disclaimer

This project is intended strictly for academic and research purposes only.
It is **not** intended for clinical use or real-world medical diagnosis.
