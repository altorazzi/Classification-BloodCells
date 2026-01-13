# Blood Cell Classification via MobileNetV3 & Test-Time Augmentation

## Project Overview
Developed a lightweight Deep Learning pipeline to classify **13,000 microscopic images** of blood cells into 8 distinct categories (e.g., Basophils, Lymphocytes, Platelets). 

The challenge involved handling a highly imbalanced dataset with significant noise (duplicates, outliers) and achieving high generalization on a hidden test set. The final solution leverages **MobileNetV3Large** with a two-stage training strategy and **Test-Time Augmentation (TTA)**.

## Data Engineering
We prioritized data quality over model complexity, using statistical methods to clean the input stream.

* **Outlier Detection via t-SNE:** Performed dimensionality reduction (t-SNE) to visualize the dataset manifold. Identified and removed clusters of "garbage" data
* **Class Balancing:** Addressed severe imbalance by upsampling minority classes, using **Geometric Augmentation** (Rotation, Flip, Translation) rather than simple duplication to prevent overfitting.

## Model Architecture & Strategy

We adopted a **Transfer Learning** approach to maximize efficiency on limited hardware.

### 1. Backbone: MobileNetV3Large
* **Stage 1 (Transfer Learning):** Frozen backbone. Trained a custom dense head (128 units, ReLU, Dropout) using **L2 Regularization** and Adam optimizer.
* **Stage 2 (Fine-Tuning):** Unfroze the top layers, increased L2 regularization to counteract the increased model capacity and reduced learning rate.

### 2. Advanced Augmentation Pipeline
* **Training:** Implemented **RandAugment** (parallelized in pipeline), applying 2 random transformations per image.
* **Inference (TTA):** Applied **Test-Time Augmentation**. During testing, each image is augmented 5 times, and predictions are averaged.

* **Insight:** The baseline CNN suffered massive overfitting (95% Val vs 24% Test). The switch to MobileNetV3 + TTA bridged the generalization gap significantly.

## Results

| Model              | Val Acc. | Test Acc. | F1-score |
|---------------------|----------|-----------|----------|
| Custom CNN          | 95.0%    | 24%       | 95%      |
| EfficientNetB0      | 91.6%    | 56%       | 92%      |
| **MobileNetV3Large** | **97.7%** | **82%**   | **97.6%** |

* **Insight:** The baseline CNN suffered massive overfitting (95% Val vs 24% Test). The switch to MobileNetV3 + TTA bridged the generalization gap significantly.
---

## Repository Structure
- `Identification_Outliers.ipynb` → Dataset cleaning & outlier detection  
- `Classification.ipynb` → Model training, fine-tuning & evaluation  
- `Report.pdf` → Full technical report (preprocessing, methods, results, discussion)  

---

## How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/altorazzi/Classification-BloodCells.git](https://github.com/altorazzi/Classification-BloodCells.git)
