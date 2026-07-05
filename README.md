# Classification of High-Dimensional Data

Final Year Project submitted for the Bachelor of Science (Honours) in Mathematics and Statistics at Hong Kong Baptist University.

This project investigates the performance of Support Vector Machines (SVM), Random Forests (RF), and Neural Networks (NN) for classification tasks involving both synthetic and real-world datasets. The study combines simulation experiments with practical applications to examine how data dimensionality, sample size, feature characteristics, and dataset complexity influence model performance.

---

## Project Objective

Classification is a fundamental problem in machine learning and statistics. As modern datasets continue to grow in size and complexity, selecting an appropriate classification algorithm becomes increasingly important.

The objectives of this project are:

* Compare the performance of SVM, Random Forest, and Neural Networks.
* Evaluate classifier behaviour on high-dimensional data.
* Investigate the impact of feature redundancy and noise.
* Apply classification algorithms to real-world datasets.
* Provide insights into the strengths and limitations of each method.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* OpenML
* Machine Learning
* Statistical Modelling
* Cross Validation
* Feature Engineering
* Data Visualisation

---

## Classification Algorithms

### Support Vector Machine (SVM)

A margin-based classifier that performs particularly well in high-dimensional feature spaces and datasets with clear class separation.

### Random Forest (RF)

An ensemble learning method that combines multiple decision trees to improve robustness, reduce overfitting, and handle heterogeneous data.

### Neural Network (NN)

A flexible machine learning model capable of learning complex non-linear relationships, particularly effective for image and pattern recognition tasks.

---

## Datasets

### Synthetic High-Dimensional Dataset

A simulated dataset consisting of 1,000 samples and 1,000 features generated using Scikit-Learn's `make_classification()` function. The simulation study allows controlled evaluation of classifier performance in high-dimensional settings.

### Breast Cancer Dataset

A structured biomedical dataset used for binary classification of malignant and benign tumours.

### Adult Income Dataset

A heterogeneous dataset containing demographic and employment-related variables used to predict whether annual income exceeds $50,000.

### MNIST Handwritten Digits Dataset

A benchmark image classification dataset consisting of handwritten digits represented as 28×28 grayscale images.

### Genomic-Style Dataset

A high-dimensional dataset designed to represent gene-expression classification problems where the number of features greatly exceeds the number of samples.

### Text Classification Dataset

A sparse high-dimensional dataset representing document classification problems using text-based features.

---

## Repository Structure

### original_style_reconstruction.py

A reconstruction of the workflow most closely matching the original project development process and thesis experiments.

### extended_thesis_reconstruction.py

A more comprehensive reconstruction implementing additional methodologies and datasets described in the final dissertation.

### Mervin_Gurung_FYP.pdf

The final dissertation submitted for the Bachelor of Science (Honours) in Mathematics and Statistics.

---

## Key Findings

* SVM performed strongly on structured datasets with clear class separation.
* Random Forest demonstrated robustness when handling mixed and noisy features.
* Neural Networks achieved the strongest performance on image-based datasets such as MNIST.
* No single algorithm consistently outperformed all others across every dataset.
* Dataset characteristics played a larger role in model performance than algorithm complexity alone.
* High-dimensional datasets present challenges including overfitting, computational cost, and feature redundancy.

---

## Note

The original project source code was not retained after project completion. The Python scripts included in this repository are reconstructed implementations based on the methodology, datasets, experiments, and results documented in the dissertation.

---

## Author

Mervin Gurung

Bachelor of Science (Honours) in Mathematics and Statistics

Hong Kong Baptist University
