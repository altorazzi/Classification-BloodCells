# Blood Cell Image Classification (AN2DL Homework)

## Project Overview
This project was developed as part of the **Artificial Neural Networks course (Politecnico di Milano)**.  
The task was to design and train deep learning models for **multi-class classification of blood cell images**.  
The dataset contained **~13,000 RGB images (96×96 px)** across **8 different blood cell types**.  

Our goal was to build an efficient architecture capable of achieving strong performance on a hidden test set, while handling data imbalance and noisy labels.

---

## Methodology

1. **Data Preprocessing**
   - Removed duplicates and corrupted images using t-SNE visualization.
   - Outlier detection identified mislabeled or irrelevant images (e.g., distorted or non-blood-cell pictures).
   - Balanced dataset with **geometric augmentations** (rotations, flips, translations).

2. **Data Augmentation**
   - Applied **RandAugment** with 2 random transformations per image.
   - Augmented validation set for underrepresented classes.
   - Used **Test-Time Augmentation (TTA)** to improve robustness at inference.

3. **Modeling Approach**
   - **Baseline CNN** (underperformed).  
   - **EfficientNetB0** (moderate performance).  
   - **MobileNetV3Small & Large** → best-performing backbone.  
   - Final model: **MobileNetV3Large + custom dense & dropout layers**, fine-tuned end-to-end.

4. **Training Strategy**
   - Optimizer: Adam  
   - Callbacks: EarlyStopping, ReduceLROnPlateau  
   - Loss: Categorical Crossentropy with class weighting  

---

## Results

| Model              | Val Acc. | Test Acc. | F1-score |
|---------------------|----------|-----------|----------|
| Custom CNN          | 95.0%    | 24%       | 95%      |
| EfficientNetB0      | 91.6%    | 56%       | 92%      |
| **MobileNetV3Large** | **97.7%** | **82%**   | **97.6%** |

**Best model**: MobileNetV3Large with advanced augmentation, achieving **82% accuracy** on the hidden test set.

---

## Repository Structure
- `Identification_Outliers.ipynb` → Dataset cleaning & outlier detection  
- `Classification.ipynb` → Model training, fine-tuning & evaluation  
- `Report.pdf` → Full technical report (preprocessing, methods, results, discussion)  

---

## How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
