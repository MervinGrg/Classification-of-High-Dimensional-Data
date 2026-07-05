"""
Classification of High-Dimensional Data
Original Style Reconstruction

Author: Mervin Gurung
BSc (Hons) Mathematics and Statistics
Hong Kong Baptist University

This script reconstructs the core experiments described in the thesis:
1. High-dimensional simulation study
2. Breast Cancer dataset
3. Adult Income dataset
4. MNIST dataset

Algorithms:
- Support Vector Machine (SVM)
- Random Forest (RF)
- Neural Network (NN)

Evaluation:
- 5-Fold Cross Validation
- Mean Accuracy
- Standard Deviation
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import (
    make_classification,
    load_breast_cancer,
    fetch_openml
)

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ============================================================
# HELPER FUNCTION
# ============================================================

def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    return scores.mean(), scores.std()

# ============================================================
# SIMULATION STUDY
# ============================================================

print("\n==============================")
print("SIMULATION STUDY")
print("==============================")

X_sim, y_sim = make_classification(
    n_samples=1000,
    n_features=1000,
    n_informative=10,
    n_redundant=50,
    n_classes=2,
    random_state=42
)

svm = SVC(kernel="linear")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

nn = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=300,
    random_state=42
)

sim_results = {}

for name, model in {
    "SVM": svm,
    "Random Forest": rf,
    "Neural Network": nn
}.items():

    mean_acc, std_acc = evaluate_model(
        model,
        X_sim,
        y_sim
    )

    sim_results[name] = [mean_acc, std_acc]

    print(
        f"{name}: "
        f"Accuracy={mean_acc:.3f} "
        f"(±{std_acc:.3f})"
    )

# ============================================================
# BREAST CANCER DATASET
# ============================================================

print("\n==============================")
print("BREAST CANCER DATASET")
print("==============================")

bc = load_breast_cancer()

X_bc = bc.data
y_bc = bc.target

results_bc = {}

for name, model in {
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=300,
        random_state=42
    )
}.items():

    mean_acc, std_acc = evaluate_model(
        model,
        X_bc,
        y_bc
    )

    results_bc[name] = [mean_acc, std_acc]

    print(
        f"{name}: "
        f"Accuracy={mean_acc:.3f} "
        f"(±{std_acc:.3f})"
    )

# ============================================================
# ADULT INCOME DATASET
# ============================================================

print("\n==============================")
print("ADULT INCOME DATASET")
print("==============================")

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

X_adult = pd.get_dummies(
    adult.data,
    drop_first=True
)

y_adult = adult.target

results_adult = {}

for name, model in {
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(50,),
        max_iter=300,
        random_state=42
    )
}.items():

    mean_acc, std_acc = evaluate_model(
        model,
        X_adult,
        y_adult
    )

    results_adult[name] = [mean_acc, std_acc]

    print(
        f"{name}: "
        f"Accuracy={mean_acc:.3f} "
        f"(±{std_acc:.3f})"
    )

# ============================================================
# MNIST DATASET
# ============================================================

print("\n==============================")
print("MNIST DATASET")
print("==============================")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

# Use subset for reasonable runtime
X_mnist = mnist.data[:10000]
y_mnist = mnist.target[:10000]

scaler = StandardScaler()

X_mnist = scaler.fit_transform(
    X_mnist
)

results_mnist = {}

for name, model in {
    "SVM": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        random_state=42
    )
}.items():

    mean_acc, std_acc = evaluate_model(
        model,
        X_mnist,
        y_mnist
    )

    results_mnist[name] = [mean_acc, std_acc]

    print(
        f"{name}: "
        f"Accuracy={mean_acc:.3f} "
        f"(±{std_acc:.3f})"
    )

# ============================================================
# VISUALISATION
# ============================================================

datasets = [
    "Breast Cancer",
    "Adult Income",
    "MNIST"
]

svm_scores = [
    results_bc["SVM"][0],
    results_adult["SVM"][0],
    results_mnist["SVM"][0]
]

rf_scores = [
    results_bc["Random Forest"][0],
    results_adult["Random Forest"][0],
    results_mnist["Random Forest"][0]
]

nn_scores = [
    results_bc["Neural Network"][0],
    results_adult["Neural Network"][0],
    results_mnist["Neural Network"][0]
]

x = np.arange(len(datasets))
width = 0.25

plt.figure(figsize=(10,6))

plt.bar(
    x - width,
    svm_scores,
    width,
    label="SVM"
)

plt.bar(
    x,
    rf_scores,
    width,
    label="Random Forest"
)

plt.bar(
    x + width,
    nn_scores,
    width,
    label="Neural Network"
)

plt.xticks(
    x,
    datasets
)

plt.ylabel("Mean Accuracy")

plt.title(
    "Classification Performance Across Datasets"
)

plt.ylim(0.5, 1.0)

plt.legend()

plt.tight_layout()

plt.show()