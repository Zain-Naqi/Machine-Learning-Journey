# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## Introduction

t-SNE is a nonlinear dimensionality reduction technique used for visualizing high-dimensional data. It preserves local similarities between data points by converting distances into probability distributions and embedding them in a lower-dimensional space (usually 2D or 3D).

## Why Use t-SNE?
- Great for **data visualization** (especially in 2D and 3D).
- Captures **local structures** better than PCA.
- Useful for **clustering** tasks.

## How t-SNE Works
1. **Compute Pairwise Similarities in High-Dimensional Space**
   - Convert Euclidean distances into **conditional probabilities** (Gaussian distribution).
   - Given two points \( x_i \) and \( x_j \), the probability that \( x_j \) is a neighbor of \( x_i \) is:
     \[
     p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)}
     \]
   - Symmetric probability:
     \[
     p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
     \]

2. **Compute Pairwise Similarities in Low-Dimensional Space**
   - Convert distances into **q-distribution** using a **Student’s t-distribution**:
     \[
     q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
     \]
   - The Student's t-distribution with **one degree of freedom** helps avoid crowding in lower dimensions.

3. **Minimize the Difference Between Distributions**
   - Use **Kullback-Leibler (KL) divergence**:
     \[
     KL(P || Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
     \]
   - Optimize using **gradient descent** to adjust \( y_i \) and reduce the difference between \( P \) and \( Q \).

---

## Example Problem

### Given Data:
We have 3D data points:
\[
X = \{(2,3,4), (8,7,6), (1,2,1), (7,8,9)\}
\]

### Steps to Apply t-SNE:
1. Compute pairwise distances and convert them into probability distributions.
2. Initialize low-dimensional points randomly in 2D.
3. Compute pairwise distances in 2D and convert them into probability distributions.
4. Compute KL divergence and update 2D positions iteratively using gradient descent.

### Solution in Python:
```python
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Sample 3D data
X = np.array([[2,3,4], [8,7,6], [1,2,1], [7,8,9]])

# Apply t-SNE to reduce to 2D
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_embedded = tsne.fit_transform(X)

# Plot the results
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.title('t-SNE Visualization')
plt.show()
```

---

## Key Parameters in t-SNE
- **Perplexity**: Controls how many nearest neighbors influence the embedding. (Typical values: **5 to 50**)
- **Learning rate**: Controls step size in gradient descent (too high → divergence, too low → slow convergence).
- **Number of iterations**: More iterations lead to better convergence but take longer.

---

## Advantages of t-SNE
✔ Preserves local structure well.
✔ Effective for high-dimensional data visualization.
✔ Works well for **clustering tasks**.

## Disadvantages of t-SNE
❌ Computationally expensive (not scalable for very large datasets).
❌ Non-deterministic (results can vary with different runs).
❌ Hard to interpret distances globally.

---

## Conclusion
- t-SNE is **great for visualization** of complex data but not ideal for general dimensionality reduction.
- It works best for **small datasets** where local similarities matter.
- Always **experiment with hyperparameters** to get the best results.

---

This document is ready to be uploaded to GitHub as `tSNE_Notes.md`. 🚀

