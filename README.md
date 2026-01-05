# Chest X-Ray Pneumonia Classification (Transfer Learning)

## Overview

This project implements a **binary image classification system** to detect **Pneumonia vs Normal** cases from chest X-ray images using **deep learning and transfer learning**.
The core objective is **not just accuracy**, but a **comparative evaluation of fine-tuning strategies** in CNN-based medical imaging models.

Three strategies are evaluated:

1. **Frozen Feature Extractor**
2. **Partial Fine-Tuning**
3. **Full Fine-Tuning**

The project is implemented entirely in **PyTorch** and structured as a reproducible experimental pipeline.

---

## Dataset

* **Dataset**: Chest X-Ray Images (Pneumonia)
* **Source**: Kaggle
* **Structure**:

  ```
  chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  └── test/
  ```

⚠️ **Important**:
This dataset is **class-imbalanced** (Pneumonia ≫ Normal). The project explicitly addresses this using **class-weighted loss**, which is not optional in medical classification.

---

## Methodology

### Model Architecture

* **Base Model**: Pretrained CNN from `torchvision.models` (ResNet family)
* **Loss Function**: CrossEntropyLoss with **class weights**
* **Optimizer**: Adam
* **Evaluation Metrics**:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-score

---

### Fine-Tuning Strategies Compared

#### 1. Frozen Feature Extractor

* Backbone weights frozen
* Only classifier head trained
* Fast, but limited representation learning

#### 2. Partial Fine-Tuning

* Early layers frozen
* Deeper layers + classifier trainable
* Best balance between generalization and stability

#### 3. Full Fine-Tuning

* Entire network trainable
* Highest capacity, highest overfitting risk

This comparison is the **actual value of the project**. Without it, this would be a generic Kaggle notebook.

---

## Data Preprocessing

* Resize to model input dimensions
* Normalization using **ImageNet mean & std**
* Conversion to tensor
* Separate transforms for training and evaluation

---

## Training Pipeline

* Custom `Dataset` class for structured loading
* PyTorch `DataLoader` with batching and shuffling
* Per-epoch tracking of:

  * Training loss
  * Validation loss
  * Training accuracy
  * Validation accuracy

---

## Results

The project produces:

* Learning curves (loss & accuracy)
* Confusion matrices
* Classification reports
* Strategy-wise comparison plots

**Key Insight**:

> Partial fine-tuning consistently provides the best trade-off between performance and stability on this dataset.

If your results don’t show this, something is wrong,either leakage, wrong freezing, or incorrect evaluation.

---

## How to Run

### Requirements

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas pillow
```

### Execution

1. Download dataset from Kaggle
2. Update dataset path:

   ```python
   base_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray"
   ```
3. Run notebook cells sequentially

---

## Limitations (Read This)

* No cross-validation (single split ≠ robust medical inference)
* Accuracy alone is **not sufficient** for clinical relevance
* Dataset quality is variable (real-world noise not handled)
* Not deployable without calibration and external validation

Pretending otherwise would be dishonest.

---

## Future Improvements

* ROC-AUC and PR-AUC metrics
* Grad-CAM for interpretability
* Cross-dataset validation
* Calibration (Platt scaling / temperature scaling)
* Clinical decision threshold optimization


