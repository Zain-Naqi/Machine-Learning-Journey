# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## Introduction

t-SNE is a dimensionality reduction technique used primarily for **visualizing high-dimensional data**. Unlike PCA, which is linear, t-SNE captures **non-linear relationships** and preserves local structures.

## Intuition

t-SNE converts high-dimensional Euclidean distances between data points into conditional probabilities representing **similarities**.

- In **high-dimensional space**, the similarity between two points \( x_i \) and \( x_j \) is given by a Gaussian distribution:
  $$
  p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma^2)}
  $$

- In **low-dimensional space**, the similarity is modeled using a **Student’s t-distribution** with one degree of freedom (heavy-tailed):
  $$
  q_{j|i} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq i} (1 + ||y_i - y_k||^2)^{-1}}
  $$

## Cost Function (KL Divergence)

t-SNE minimizes the **Kullback-Leibler (KL) divergence** between the probability distributions in high and low-dimensional spaces:
  $$
  C = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}
  $$

where \( p_{ij} = \frac{p_{j|i} + p_{i|j}}{2} \) and \( q_{ij} = \frac{q_{j|i} + q_{i|j}}{2} \).

## Algorithm Steps
1. Compute pairwise similarities \( p_{ij} \) in high-dimensional space using Gaussian distributions.
2. Initialize random **low-dimensional** embeddings \( y_i \).
3. Compute pairwise similarities \( q_{ij} \) in low-dimensional space using **Student’s t-distribution**.
4. Minimize KL divergence via **gradient descent**.
5. Update low-dimensional embeddings iteratively.

## Example Problem
### Given:
A dataset with three points in 3D space:

| Point | x | y | z |
|--------|----|----|----|
| A      | 2  | 3  | 5  |
| B      | 1  | 1  | 2  |
| C      | 4  | 5  | 6  |

### Solution:
1. Compute pairwise Euclidean distances:
   - \( d(A, B) = \sqrt{(2-1)^2 + (3-1)^2 + (5-2)^2} = \sqrt{1+4+9} = \sqrt{14} \)
   - \( d(A, C) = \sqrt{(2-4)^2 + (3-5)^2 + (5-6)^2} = \sqrt{4+4+1} = \sqrt{9} = 3 \)
   - \( d(B, C) = \sqrt{(1-4)^2 + (1-5)^2 + (2-6)^2} = \sqrt{9+16+16} = \sqrt{41} \)

2. Convert distances into probability distributions \( p_{ij} \).
3. Map the points to 2D space using t-SNE.

## Advantages
✅ Preserves local structure better than PCA.
✅ Captures **non-linear relationships**.
✅ Works well for **high-dimensional** data visualization.

## Disadvantages
❌ Computationally expensive (complexity: \( O(N^2) \)).
❌ Results depend on hyperparameters (**perplexity**).

## Key Hyperparameters
- **Perplexity**: Controls balance between local and global structure (default ~30).
- **Learning Rate**: Affects convergence.
- **Iterations**: More iterations lead to better separation.

## Summary
- **t-SNE** is a powerful tool for visualizing high-dimensional data.
- It captures **local neighborhoods** more effectively than PCA.
- Requires careful tuning of **perplexity** and learning rate.

## Implementation in Python
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Sample high-dimensional data
X = np.random.rand(100, 5)  # 100 points in 5D

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_embedded = tsne.fit_transform(X)

plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.title("t-SNE Visualization")
plt.show()
```
