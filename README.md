# 🧠 Deep Learning for Perception — Empirical Study of Neural Architecture, Optimization, and Regularization Dynamics

> **Building, Breaking, and Optimizing Multi-Layer Perceptrons on Visual Data**

## 📌 Project Overview

This repository presents an empirical study on neural network dynamics, optimization, and generalization for computer vision tasks using the **Fashion-MNIST** benchmark dataset.

The project investigates a full end-to-end deep learning pipeline:

1. **Manual backpropagation** implementation using NumPy from first principles.

2. **Analytical gradient verification** against PyTorch autograd.

3. **Activation function behavior** and gradient flow analysis across depth.

4. **Classification vs. regression** loss function formulations.

5. **First-order optimizer convergence dynamics** under fixed and tuned learning rates.

6. **Overfitting diagnosis** and structural bias-variance tradeoff analysis.

7. **Regularization techniques** and capacity control mechanisms.

8. **Automated hyperparameter optimization** via 5-fold stratified cross-validation.

9. **Unbiased held-out test set evaluation** and comprehensive error diagnostics.

The primary objective is to evaluate how architecture design, initialization, optimization algorithms, and regularization strategies impact loss landscapes, learning dynamics, and model generalization.

## 🎯 Core Objectives

* **First-Principles Modeling:** Derive and implement forward and backward propagation from scratch in pure NumPy.

* **Gradient Verification:** Compute exact analytical gradients and verify numerical precision against PyTorch automatic differentiation.

* **Activation Dynamics:** Benchmark Sigmoid, Tanh, ReLU, and Leaky ReLU (tracking gradient magnitudes and dead units).

* **Loss Dynamics:** Compare Categorical Cross-Entropy against Mean Squared Error (MSE) for multi-class visual recognition, alongside multi-metric regression formulations (MSE, RMSE, MAE).

* **Optimization Benchmarking:** Analyze convergence rates for SGD, SGD with Momentum, RMSProp, and Adam.

* **Capacity Control & Overfitting:** Deliberately induce over-parameterization on restricted training sets to diagnose the generalization gap.

* **Empirical Regularization:** Evaluate $L_1$/$L_2$ decay, Dropout, Batch Normalization, Data Augmentation, and Early Stopping.

* **Hyperparameter Optimization:** Perform automated randomized search over structural and optimization hyperparameter spaces using 5-fold stratified cross-validation.

* **Final Diagnostics:** Evaluate final generalized performance using Macro Precision, Recall, F1-Score, and Confusion Matrix visualization.

## 🗂️ Dataset Benchmark & Split Distribution

### Fashion-MNIST Dataset

Each sample is a grayscale $28 \times 28$ image, flattened into a 784-dimensional input vector and normalized to $[0, 1]$.

| Label | Class Name | Description | Train Count | Validation Count | Test Count | 
 | ----- | ----- | ----- | ----- | ----- | ----- | 
| 0 | T-shirt/top | Upper body wear | 800 | 200 | 6,000 | 
| 1 | Trouser | Lower body wear | 800 | 200 | 6,000 | 
| 2 | Pullover | Upper body wear | 800 | 200 | 6,000 | 
| 3 | Dress | Full body wear | 800 | 200 | 6,000 | 
| 4 | Coat | Outerwear | 800 | 200 | 6,000 | 
| 5 | Sandal | Footwear | 800 | 200 | 6,000 | 
| 6 | Shirt | Upper body wear | 800 | 200 | 6,000 | 
| 7 | Sneaker | Footwear | 800 | 200 | 6,000 | 
| 8 | Bag | Accessory | 800 | 200 | 6,000 | 
| 9 | Ankle boot | Footwear | 800 | 200 | 6,000 | 
| **Total** | — | — | **8,000** | **2,000** | **60,000** | 

*Dataset Source:* [Zalando Research / Kaggle Fashion-MNIST Benchmark](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## 💻 Execution Environment & Reproducibility

* **Platform:** Kaggle Notebook Environment

* **PyTorch Version:** `2.10.0+cu128`

* **Hardware Accelerator:** NVIDIA GPU Tesla T4 (CUDA Active)

* **Primary Libraries:** `NumPy`, `Pandas`, `Matplotlib`, `PyTorch`, `torchvision`, `scikit-learn`

* **Global Seed:** `42`

```
import numpy as np
import torch
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

```

Executing the notebook sequentially guarantees exact numerical reproducibility across all empirical trials.

# 📚 Experimental Phases & Results

## Phase 1 — First-Principles Backpropagation

A zero-dependency Multi-Layer Perceptron (MLP) implemented in NumPy:

$$
\text{Input } (784) \longrightarrow \text{Hidden Layer } (64, \text{ReLU}) \longrightarrow \text{Output Layer } (10, \text{Softmax})
$$

* **Loss Function:** Categorical Cross-Entropy

* **Training Subset:** 5,000 samples | Batch Size: 128 | Learning Rate: 0.1 | 20 Epochs

* **Verification:** Analytical gradients verified against PyTorch automatic differentiation with max absolute discrepancy $< 10^{-7}$.

### NumPy MLP Loss Trajectory (20 Epochs)

| Epoch | Loss | Epoch | Loss | 
 | ----- | ----- | ----- | ----- | 
| 01 | 2.1346 | 11 | 0.5830 | 
| 02 | 1.3766 | 12 | 0.5976 | 
| 03 | 1.0299 | 13 | 0.5481 | 
| 04 | 0.8763 | 14 | 0.5459 | 
| 05 | 0.7868 | 15 | 0.5196 | 
| 06 | 0.7467 | 16 | 0.5035 | 
| 07 | 0.7036 | 17 | 0.5354 | 
| 08 | 0.6868 | 18 | 0.4974 | 
| 09 | 0.6214 | 19 | 0.4701 | 
| 10 | 0.6329 | 20 | **0.4755** | 

## Phase 2 — Activation Dynamics & Gradient Flow

Comparative analysis across Sigmoid, Tanh, ReLU, and Leaky ReLU activations:

* Tracking validation loss trajectories over training epochs.

* Monitoring first-hidden-layer gradient norms ($\Vert{}\nabla_{W_1}\Vert{}$) to detect vanishing/exploding gradients.

* Quantifying the proportion of "dead" ReLU units ($\text{activation} = 0$) across validation batches.

## Phase 3 — Loss Function Dynamics

* **Classification Formulations:** Categorical Cross-Entropy vs. One-Hot Encoded Mean Squared Error (MSE).

* **Regression Formulation:** Continuous target regression evaluating MSE, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

## Phase 4 — Optimization Algorithms

Comparative convergence benchmarking across four first-order optimizers:

1. Standard Stochastic Gradient Descent (SGD)

2. SGD + Nesterov / Classical Momentum

3. RMSProp

4. Adam

Evaluated under both uniform baseline learning rates ($\eta = 0.001$) and independently tuned learning rates. Tracking wall-clock time, final validation accuracy, and epochs required to hit target threshold ($\ge 85\%$).

## Phase 5 — Capacity Control & Overfitting Diagnosis

Deliberate over-parameterization induced by restricting training data to $N = 2,000$ samples while expanding network architecture to $\ge 4$ hidden layers with 512 units each.

Training proceeds until training accuracy reaches $> 99\%$, measuring loss saturation and the resulting generalization gap ($\text{Acc}_{\text{train}} - \text{Acc}_{\text{val}}$).

## Phase 6 — Regularization Techniques

Mitigating high variance from Phase 5 by systematically evaluating individual and combined techniques:

* $L_2$ **Weight Decay:** Penalty factor evaluation across logarithmic scales ($\lambda \in [10^{-4}, 10^{-1}]$).

* $L_1$ **Sparsity Penalty:** Tracking weight pruning percentages ($\vert{}w\vert{} < 10^{-3}$).

* **Dropout:** Testing drop probabilities ($p \in [0.1, 0.5]$).

* **Batch Normalization:** Internal covariate shift control applied pre/post-activation.

* **Early Stopping:** Patience-based termination triggered by validation loss divergence.

* **Data Augmentation:** Random horizontal flips and subtle random rotations ($\pm 10^\circ$).

* **Dataset Scaling:** Expanding dataset size from 2,000 to 10,000 and 20,000 training samples.

## Phase 7 — Automated Hyperparameter Tuning & Cross-Validation

Randomized grid search over parameter space ($N \ge 12$ configurations):

* Learning Rate ($\eta$)

* Hidden Layer Width

* Dropout Probability ($p$)

Evaluated using **5-Fold Stratified Cross-Validation**. Models are ranked by mean cross-validation accuracy and score variance across folds.

## 📊 Final Model Evaluation & Empirical Results

The top-performing architecture from Phase 7 is retrained on the combined dataset and evaluated against the held-out test set.

### Optimal Hyperparameter Configuration

| Parameter | Selected Value | 
 | ----- | ----- | 
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | 
| **Learning Rate (**$\eta$**)** | $0.001$ | 
| **Hidden Layer Architecture** | $[256, 128]$ (2 Hidden Layers) | 
| **Activation Function** | Leaky ReLU ($\alpha=0.01$) | 
| **Dropout Probability (**$p$**)** | $0.20$ | 
| $L_2$ **Weight Decay (**$\lambda$**)** | $10^{-4}$ | 
| **Batch Normalization** | Enabled (Post-Linear) | 
| **Total Epochs Trained** | 35 (Early Stopping trigger) | 

### Held-Out Test Set Performance Metrics

| Metric | Result | 
 | ----- | ----- | 
| **Test Accuracy** | **88.65%** | 
| **Macro Precision** | **0.8872** | 
| **Macro Recall** | **0.8865** | 
| **Macro F1-Score** | **0.8860** | 
| **Net Gain over Phase 2 Baseline** | **+6.45 pp** | 

### Key Diagnostic Findings

1. **Activation Stability:** ReLU and Leaky ReLU significantly outperformed Sigmoid/Tanh by mitigating vanishing gradient problems in deep architectures.

2. **Optimizer Efficiency:** Adam achieved target convergence ($>85\%$ validation accuracy) in significantly fewer epochs compared to standard SGD.

3. **Variance Reduction:** Combining Batch Normalization with $p=0.2$ Dropout reduced the generalization gap from $14.2\%$ (overfitted baseline) down to $< 2.5\%$.

## ⚠️ Evaluation Protocol & Data Integrity

To eliminate data leakage and optimism bias, the held-out test set was isolated completely until the final evaluation phase. All architectural design decisions, hyperparameter selection, and regularizer comparisons relied strictly on training/validation splits and 5-fold cross-validation scores.

## 🧪 Setup & Execution Instructions

### 1. Kaggle Environment Setup

Create a new Kaggle Notebook and set the hardware accelerator to **GPU Tesla T4**.

### 2. Dataset Attachment

Attach the Zalando Research `fashionmnist` dataset to your working directory.

### 3. Execution

Run the complete notebook sequentially:

```
DLP_NN_Optimization_Study.ipynb

```

## 📁 Repository Structure

```
DLP-Neural-Network-Optimization/
│
├── DLP-project.ipynb    # Complete Kaggle notebook with outputs
├── README.md                           # Comprehensive empirical study report
└── Research_Summary_Report.docx        # Formatted project documentation

```
