Neural networks in machine learning are based on **mathematical formulas** for linear algebra and calculus, especially for forward propagation and backpropagation.

Here’s a structured summary of the **core neural network formulas**:

---

### 🧠 1. **Forward Propagation**

For each layer `l` in the network:

#### **a. Linear transformation:**

$$
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
$$

* $Z^{[l]}$: Pre-activation value (vector)
* $W^{[l]}$: Weights matrix of layer `l`
* $A^{[l-1]}$: Activations from previous layer
* $b^{[l]}$: Bias vector

#### **b. Activation:**

$$
A^{[l]} = g(Z^{[l]})
$$

* $g$: Activation function (e.g., sigmoid, ReLU, tanh)

If $l = 1$, then:

$$
A^{[0]} = X \quad \text{(input features)}
$$

---

### 🧠 2. **Common Activation Functions**

| Function | Formula                                                | Range            |
| -------- | ------------------------------------------------------ | ---------------- |
| Sigmoid  | $\sigma(z) = \frac{1}{1 + e^{-z}}$                     | (0, 1)           |
| Tanh     | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$         | (−1, 1)          |
| ReLU     | $\text{ReLU}(z) = \max(0, z)$                          | \[0, ∞)          |
| Softmax  | $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | \[0, 1], sum = 1 |

---

### 🧠 3. **Loss Functions**

#### **a. Binary Cross Entropy (for binary classification):**

$$
\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

#### **b. Mean Squared Error (for regression):**

$$
\mathcal{L} = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2
$$

---

### 🧠 4. **Backpropagation (Gradient Computation)**

This step updates weights using derivatives of the loss.

#### **a. Compute gradient of loss w\.r.t. output:**

$$
\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial A^{[L]}} \cdot g'(Z^{[L]})
$$

#### **b. Backpropagate through layers:**

$$
\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \cdot g'(Z^{[l]})
$$

#### **c. Compute gradients:**

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} \delta^{[l]} (A^{[l-1]})^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \frac{1}{m} \sum \delta^{[l]}
$$

---

### 🧠 5. **Gradient Descent Weight Update**

$$
W^{[l]} = W^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial W^{[l]}}
$$

$$
b^{[l]} = b^{[l]} - \alpha \frac{\partial \mathcal{L}}{\partial b^{[l]}}
$$

* $\alpha$: Learning rate

---

### Optional: Matrix View of Entire Network

Let’s say you have:

* $X \in \mathbb{R}^{n_x \times m}$ (input features)
* Multiple layers with sizes $n^{[1]}, n^{[2]}, ..., n^{[L]}$

You can write all computations in terms of matrix products and activations.

---