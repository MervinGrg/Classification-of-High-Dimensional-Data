
"""
Classification of High-Dimensional Data (Reconstructed Final FYP Script)
Author: Mervin Gurung
Degree: BSc (Hons) Mathematics and Statistics
HKBU

This script reconstructs the full experimental pipeline described in the thesis:
- Simulation study (high-dimensional synthetic data)
- Real-world datasets (Breast Cancer, Adult Income, MNIST)
- High-dimensional genomic-style dataset
- Text classification dataset (TF-IDF + Chi-square)
- Models: SVM, Random Forest, Neural Network
- Evaluation: 5-fold cross-validation (accuracy mean ± std)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, fetch_openml, fetch_20newsgroups, make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ----------------------------
# Evaluation function
# ----------------------------
def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


# ----------------------------
# Models
# ----------------------------
svm = SVC(kernel='linear')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
nn = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)


# ============================
# 1. SIMULATION STUDY
# ============================
print("Running Simulation Study...")

X_sim, y_sim = make_classification(
    n_samples=1000,
    n_features=1000,
    n_informative=10,
    n_redundant=200,
    random_state=42
)

sim_results = {
    "SVM": evaluate_model(svm, X_sim, y_sim),
    "Random Forest": evaluate_model(rf, X_sim, y_sim),
    "Neural Network": evaluate_model(nn, X_sim, y_sim)
}

print("Simulation Results:", sim_results)


# ============================
# 2. BREAST CANCER
# ============================
print("Running Breast Cancer...")

bc = load_breast_cancer()
X_bc, y_bc = bc.data, bc.target

bc_results = {
    "SVM": evaluate_model(svm, X_bc, y_bc),
    "Random Forest": evaluate_model(rf, X_bc, y_bc),
    "Neural Network": evaluate_model(nn, X_bc, y_bc)
}


# ============================
# 3. ADULT INCOME
# ============================
print("Running Adult Income...")

adult = fetch_openml(name="adult", version=2, as_frame=True)
X_adult = adult.data
y_adult = adult.target

X_adult = pd.get_dummies(X_adult).fillna(0)

adult_results = {
    "SVM": evaluate_model(svm, X_adult, y_adult),
    "Random Forest": evaluate_model(rf, X_adult, y_adult),
    "Neural Network": evaluate_model(nn, X_adult, y_adult)
}


# ============================
# 4. MNIST
# ============================
print("Running MNIST...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X_mnist, y_mnist = mnist.data, mnist.target.astype(int)

scaler = StandardScaler()
X_mnist = scaler.fit_transform(X_mnist)

mnist_results = {
    "SVM": evaluate_model(svm, X_mnist, y_mnist),
    "Random Forest": evaluate_model(rf, X_mnist, y_mnist),
    "Neural Network": evaluate_model(nn, X_mnist, y_mnist)
}


# ============================
# 5. GENOMIC DATASET (HIGH-DIMENSIONAL)
# ============================
print("Running Genomic Dataset...")

X_gen, y_gen = make_classification(
    n_samples=200,
    n_features=1000,
    n_informative=20,
    random_state=42
)

X_gen = StandardScaler().fit_transform(X_gen)
X_gen = PCA(n_components=50).fit_transform(X_gen)

gen_results = {
    "SVM": evaluate_model(svm, X_gen, y_gen),
    "Random Forest": evaluate_model(rf, X_gen, y_gen),
    "Neural Network": evaluate_model(nn, X_gen, y_gen)
}


# ============================
# 6. TEXT CLASSIFICATION
# ============================
print("Running Text Classification...")

news = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
X_text, y_text = news.data, news.target

tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
X_text = tfidf.fit_transform(X_text)

X_text = SelectKBest(chi2, k=5000).fit_transform(X_text, y_text)

text_results = {
    "SVM": evaluate_model(svm, X_text, y_text),
    "Random Forest": evaluate_model(rf, X_text, y_text),
    "Neural Network": evaluate_model(nn, X_text, y_text)
}


# ============================
# COLLECT ALL RESULTS
# ============================
datasets = {
    "Simulation": sim_results,
    "Breast Cancer": bc_results,
    "Adult Income": adult_results,
    "MNIST": mnist_results,
    "Genomic": gen_results,
    "Text": text_results
}


# ============================
# PRINT SUMMARY
# ============================
print("\nFINAL RESULTS SUMMARY:\n")

for dname, res in datasets.items():
    print(f"\n{dname}")
    for model, (mean, std) in res.items():
        print(f"{model}: {mean:.3f} ± {std:.3f}")


# ============================
# VISUALISATION
# ============================
print("\nGenerating plots...")

# --- Simulation plot ---
plt.figure(figsize=(8,5))
labels = list(sim_results.keys())
means = [sim_results[k][0] for k in labels]
stds = [sim_results[k][1] for k in labels]

plt.bar(labels, means, yerr=stds, capsize=5)
plt.title("Simulation Study Results")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# --- Real-world comparison ---
real_labels = list(bc_results.keys())

real_datasets = ["Breast Cancer", "Adult", "MNIST", "Genomic", "Text"]

X = np.arange(len(real_labels))
width = 0.15

plt.figure(figsize=(12,6))

for i, (dname, res) in enumerate([
    ("Breast Cancer", bc_results),
    ("Adult Income", adult_results),
    ("MNIST", mnist_results),
    ("Genomic", gen_results),
    ("Text", text_results)
]):
    means = [res[k][0] for k in real_labels]
    plt.bar(X + i*width, means, width, label=dname)

plt.xticks(X + width*2, real_labels)
plt.ylabel("Accuracy")
plt.title("Model Performance Across Datasets")
plt.legend()
plt.tight_layout()
plt.show()
