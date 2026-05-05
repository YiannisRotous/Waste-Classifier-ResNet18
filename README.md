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
To evaluate the impact of model depth on classification accuracy, we compared a baseline **18-layer architecture** against a deeper **50-layer variant**. The primary bottleneck for the 18-layer model was the confusion between **Plastic** and **White-Glass**.

* **18-Layer Model:** Misclassified **12 instances** of Plastic as White-Glass.
* **50-Layer Model:** Reduced this confusion to **7 instances**, representing a **~42% improvement** in distinguishing between these two transparent/reflective materials.

<table>
  <tr>
    <td><b>18-Layer Confusion Matrix</b></td>
    <td><b>50-Layer Confusion Matrix</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/21b3f632-37d0-4050-90ef-20e463495470" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/8ebaae55-3836-45e6-9ae2-93dd0750062c" width="400"></td>
  </tr>
</table>


### **Key Observations:**

* **High Confidence:** Both models show exceptional precision in distinct categories. The **50-layer model** maintains high accuracy for **Batteries** (89) and **Clothes** (522), showing virtually no leakage into unrelated categories, which validates the model's ability to handle high-volume classes.
* **Minor Confusion:** While the **50-layer model** significantly improved material detection, a slight visual overlap between **Plastic** and **White Glass** remains the most common error. This is expected given their nearly identical transparency and reflective properties under varied lighting conditions.
* **Textile Distinction:** The model successfully distinguishes between **Clothes** and **Shoes** with near-perfect accuracy (improving Shoe detection from 187 to 190 in the 50-layer model). This justifies the decision to split textiles into two separate categories, as the deeper model captures the textural differences effectively.
* **Reflective Material Gains:** The deeper architecture showed notable gains in identifying **Metal** (increasing from 64 to 69) and **White-Glass** (increasing from 65 to 72). This suggests that the additional layers allow the network to learn the subtle gradients and edge definitions unique to reflective surfaces.

### 2. Category Accuracy Leaderboard
The primary performance leap for the **50-layer model** was observed in the most challenging categories, specifically **Plastic** and **White-Glass**, which previously acted as the "weakest links" in the baseline model.

* **18-Layer Model:** Struggled significantly with **Plastic**, recording the lowest accuracy at **70.9%**.
* **50-Layer Model:** Boosted **Plastic** accuracy to **82.6%**, representing a massive **+11.7% gain** through deeper feature extraction.
  
<table>
  <tr>
    <td><b>18-Layer Accuracy by Category</b></td>
    <td><b>50-Layer Accuracy by Category</b></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d6820657-f7db-444b-8fe2-0c35441985fc" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/42cd810f-6c92-44f4-ae02-e251556cc13b" width="400"></td>
  </tr>
</table>


**Key Observations:**

* **Top-Tier Consistency:** Both models excel at classifying **Biological** waste and **Trash**, but the 50-layer model pushes **Biological** performance to a near-perfect **99.0%**.
* **Resolution of Ambiguity:** The 50-layer model shows a significant increase in **White-glass** (+9.1%) and **Metal** (+6.6%) accuracy. This confirms that the additional depth allows the network to better interpret the complex light patterns found on reflective and transparent surfaces.
* **Improved Glass Sorting:** **Brown-glass** accuracy saw a substantial rise from **90.0% to 96.7%**. This suggests the deeper architecture is much more efficient at distinguishing color-specific glass textures from other transparent contaminants.
* **Diminishing Returns on Simple Textures:** Categories with distinct, non-reflective textures like **Clothes** and **Green-glass** remained stable across both architectures. This indicates that while the 50-layer model is essential for difficult materials, the 18-layer model is already highly capable of identifying textiles and colored glass.


### 📈 Global Performance Summary
| Metric | 18-Layer Model | 50-Layer Model | Delta |
| :--- | :--- | :--- | :--- |
| **Final Test Accuracy** | **92.63%** | **94.76%** | **+2.13%** |
| **Total Images Tested** | 1,557 | 1,557 | — |
| **Best Performing Class** | Batteries / Shoes | Biological / Trash | — |
| **Most Complex Class** | Plastic (70.9%) | Plastic (82.6%) | **+11.7%** |

### ⏱️ Computational Efficiency
*Testing conducted on an **NVIDIA T4 GPU** (Google Colab).*

| Metric | 18-Layer (ResNet-18) | 50-Layer (ResNet-50) | Increase |
| :--- | :--- | :--- | :--- |
| **Time per Epoch** | ~70 seconds | 154 seconds | **+120%** |
| **Total Training Time** | ~58 minutes | ~128 minutes | **+70 mins** |
| **Epochs** | 50 | 50 | — |

### 💡 Key Takeaway
While the **50-layer model** requires more than double the training time per epoch, it is the superior choice for this application. The **2.13% gain** in overall accuracy is driven primarily by its success in the "Most Complex Class" (**Plastic**), where it improved reliability by nearly **12%**. For industrial recycling tasks, this reduction in misclassification far outweighs the additional computational cost.
