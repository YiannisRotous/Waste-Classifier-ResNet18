# Waste-Classifier-ResNet18
Addressing the growing challenge of household waste management through deep learning. This project compares two Deep learning architectures (ResNet18 vs. ResNet50) to classify 12 household waste categories, evaluating their accuracy and inference efficiency to provide actionable insights for environmental statistics and policy-making.


# ♻️ Household Garbage Classification (12 Categories)

We have developed a Deep Learning algorithm to classify household waste into 12 distinct categories using **PyTorch** and **Transfer Learning**. This project specifically addresses the inherent class imbalances within the dataset to provide a robust classification. We provide a classification system enabling better sorting for recycling processes, an environmental question that still percists and it is of quite improtance for policy-makers. 

## 📂 Dataset
The model is trained on the [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by Mostafa Mohamed. It contains over 15,000 images categorized into:
* **Paper & Cardboard**
* **Glass:** Green, Brown, White
* **Metal & Plastic**
* **Textiles:** Clothes, Shoes
* **Hazardous:** Batteries
* **Biological:** Organic waste
* **Trash:** Non-recyclable

## 🛠️ Tech Stack
We evaluated two architectures—ResNet18 and ResNet50—to analyze the trade-offs between model complexity and classification accuracy versus computational cost. By comparing these two versions, we examined whether the deeper architecture of ResNet50 yielded significant performance gains or if the increased training time outweighed the benefits. The project was implemented using PyTorch within the Google Colab environment, leveraging an NVIDIA T4 GPU to accelerate the computationally intensive training process.

## 🚀 Model Training
The training phase was designed as a comparative study between two architectures: **ResNet18** and **ResNet50**. This allowed for a detailed analysis of the trade-offs between model depth and computational efficiency, determining if the added complexity of a 50-layer network provided significant accuracy gains to justify the increased training time. We used the standard **80/10/10 split** for training, validation, and testing, respectively. The models were optimized using the **Adam optimizer** and **CrossEntropyLoss**, a standard criterion for multi-class classification tasks. The training loop was implemented with integrated performance monitoring, including real-time loss calculation and accuracy tracking across all 50 epochs.

### Key Features:
* **Comprehensive Dataset Auditing:** Before training, the pipeline performs a full audit of the 15,515 images, verifying dimensions, RGB color channels, and aspect ratios to ensure data integrity.
* **Targeted Class Imbalance Handling:** The project specifically addresses the inherent imbalances within the 12 garbage categories, tailoring the training process to ensure the model remains robust and unbiased toward majority classes.
* **Granular 12-Class Classification:** Moving beyond standard 6-class waste models, this project utilizes a high-granularity system (including specific glass colors and textile types) to provide more actionable data for modern recycling facilities.
* **Automated & Reproducible Preprocessing:** By utilizing the `split-folders` library with a fixed seed (1337), the project ensures a consistent data distribution. This reproducibility is critical for making valid scientific comparisons between the ResNet18 and ResNet50 experiments.

## 📊 Results & Analysis

### 1. Confusion Matrix
The confusion matrix highlights the high precision of the model, with a very strong diagonal line.

<img width="997" height="925" alt="ConfMat18" src="https://github.com/user-attachments/assets/21b3f632-37d0-4050-90ef-20e463495470" />

**Key Observations:**
* **High Confidence:** Most classes show very little leakage, especially distinct categories like **Batteries** and **Clothes**.
* **Minor Confusion:** There is a slight visual overlap between **Plastic** and **White Glass**, which is expected given their similar transparency and reflective properties under certain lighting conditions.
* **Textile Distinction:** The model successfully distinguishes between **Clothes** and **Shoes** with near-perfect accuracy, justifying the decision to split textiles into two categories.

### 2. Category Accuracy Leaderboard
This chart ranks the classes from highest to lowest accuracy on the test set.

<img width="989" height="790" alt="Sorted18" src="https://github.com/user-attachments/assets/d6820657-f7db-444b-8fe2-0c35441985fc" />

**Analysis:**
* **Top Performers:** Categories like **Batteries**, **Shoes**, and **Green Glass** achieved nearly 100% accuracy. These items have highly consistent shapes and colors that the ResNet18 feature extractors identified easily.
* **Challenges:** **Plastic** and **Paper** appear slightly lower on the leaderboard (though still remarkably high). This is likely due to the high "intra-class variance"—for example, paper can be a flat sheet, a crumpled ball, or a shredded pile, making it harder for the model to find a single "canonical" shape.


## 📈 Summary Table
| Metric | Result |
| :--- | :--- |
| **Final Test Accuracy** | **92.63%** |
| **Total Images Tested** | 1,557 |
| **Best Performing Class** | Batteries / Shoes |
| **Most Complex Class** | Plastic |

## ⏱️ Computational Efficiency (ResNet18)
| Metric | Value |
| :--- | :--- |
| **Time per Epoch** | ~70 seconds |
| **Total Training Time** | ~58 minutes (50 Epochs) |
| **Hardware** | NVIDIA T4 GPU (Google Colab) |
| **Official Test Accuracy** | 92.63% |
