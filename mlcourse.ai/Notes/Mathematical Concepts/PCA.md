# **Principal Component Analysis (PCA) – Detailed Notes**

## **1. Introduction**
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while preserving as much variance as possible.

### **Why Use PCA?**
- High-dimensional data can be difficult to analyze and visualize.
- Some features may be redundant or correlated.
- PCA reduces dimensions while retaining the most important patterns.

---

## **2. Intuition Behind PCA**
Imagine a dataset with multiple features (e.g., height, weight, age). PCA finds the best axes (**principal components**) to represent the data in a lower-dimensional space while keeping as much information as possible.

- The **first principal component** explains the most variance.
- The **second principal component** is orthogonal to the first and explains the second-most variance.
- This process continues for subsequent components.

---

## **3. Steps of PCA**

### **Step 1: Standardize the Data**
Since PCA is affected by scale, we standardize the data to have:
- Mean = **0**
- Variance = **1**

**Formula:**
\[
X' = \frac{X - \mu}{\sigma}
\]

where:
- \(X\) is the original data
- \(\mu\) is the mean of each feature
- \(\sigma\) is the standard deviation

**Why standardize?** If features have different scales (e.g., age in years vs. salary in dollars), the feature with the highest variance will dominate.

---

### **Step 2: Compute the Covariance Matrix**
The covariance matrix captures the relationship between features:

\[
C = \frac{1}{n} X^T X
\]

where:
- \(X\) = Standardized data matrix
- \(C\) = Covariance matrix

**Key idea:**
- **Large covariance** → Two variables are correlated.
- **Zero covariance** → Two variables are independent.

---

### **Step 3: Compute Eigenvalues and Eigenvectors**
To find the principal components, we compute eigenvalues and eigenvectors of the covariance matrix:

\[
C v = \lambda v
\]

where:
- \(C\) is the covariance matrix
- \(v\) is an eigenvector (principal component)
- \(\lambda\) is an eigenvalue (explains variance along that component)

**Key idea:**
- Eigenvectors = **Directions** (axes) of maximum variance.
- Eigenvalues = **Magnitude** of variance explained by each eigenvector.

---

### **Step 4: Select Top \( k \) Principal Components**
- Sort eigenvalues in **descending order**.
- Choose the **top \( k \) components** that capture most of the variance.

**Variance retained:**
\[
\text{Explained variance ratio} = \frac{\lambda_i}{\sum \lambda}
\]

**Choosing \( k \):** Use the **cumulative explained variance plot**:
- If **95% variance** is explained by the first **2** components, then choose \( k = 2 \).

---

### **Step 5: Project Data onto New Axes**
Transform the original data into the new reduced-dimensional space:

\[
X_{\text{new}} = X W_k
\]

where:
- \(W_k\) is the matrix of the top \( k \) eigenvectors.

---

## **4. Example: PCA in Python**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample Data (5 points, 3 features)
X = np.array([[2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9], [6, 8, 10]])

# Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (reduce from 3D to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Transformed Data:\n", X_pca)

# Explained Variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```

---

## **5. PCA Interpretation**
### **1️⃣ PCA Finds New Features**
- The original features are transformed into a new set of uncorrelated features (principal components).
- These new features maximize variance.

### **2️⃣ PCA Reduces Noise**
- By removing low-variance components, PCA helps filter out noise.

### **3️⃣ PCA for Visualization**
- PCA helps reduce high-dimensional data to 2D or 3D, making it easier to visualize.

---

## **6. When to Use PCA?**
✅ **High-dimensional datasets** (e.g., image processing, genetics)
✅ **Visualization** (reducing data to 2D or 3D for plotting)
✅ **Feature selection** (removing redundant features)
✅ **Noise reduction**

🚫 **When NOT to Use PCA?**
- If **features are not correlated**, PCA won’t be useful.
- If **data is not linear**, consider **t-SNE or UMAP**.

---

## **7. PCA vs. Other Dimensionality Reduction Methods**
| Method   | Linear/Non-linear | Best For |
|----------|----------------|---------|
| **PCA**  | Linear         | Feature extraction, compression |
| **t-SNE** | Non-linear    | Data visualization |
| **UMAP**  | Non-linear    | Better structure preservation than t-SNE |

---

## **8. Summary**
- **PCA finds new axes** (principal components) that **maximize variance**.
- It uses **eigenvalues and eigenvectors** from the **covariance matrix**.
- We select **top \( k \) components** to reduce dimensions while **preserving variance**.
- PCA is useful for **feature selection, noise reduction, and visualization**.

---

