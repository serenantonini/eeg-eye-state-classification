# EEG Eye State Classification

Classification of eye state (open vs. closed) from 14-channel EEG signals using supervised Machine Learning models.

Academic project developed for the **Machine Learning** course within the Master's Degree in Computer Science.

**Authors:** Serena Celeste Antonini · Valeria Crippa

---

## Project Description

Electroencephalography (EEG) is a non-invasive technique that measures electrical brain activity through electrodes placed on the scalp. The ability to automatically detect eye state (open/closed) from these signals is relevant to several application domains.

The goal of this project is to build, optimize, and compare supervised classification models based on different paradigms — **probabilistic** (Naive Bayes), **tree-based** (Decision Tree), and **geometric** (SVM) — evaluating their accuracy, generalization capability, and computational efficiency.

## Dataset

The project uses the **EEG Eye State Dataset**, publicly available on Kaggle, consisting of 14,980 observations with 14 continuous numerical features (signals measured in µV) collected from the following electrodes:

```text
AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
```

The target variable is binary:

| Value | Class                | % of Dataset |
| ----- | -------------------- | ------------ |
| 0     | Open (eyes open)     | 56.1%        |
| 1     | Closed (eyes closed) | 43.9%        |

The dataset provides a native Training Set (9,980 samples) / Test Set (5,000 samples) split, which is preserved throughout the entire experimental study.

## Repository Structure

```text
.
├── eeg_eye_state_analysis.ipynb   # Complete pipeline (EDA, preprocessing, models, evaluation)
├── report.pdf                     # Full technical project report
└── README.md
```

## Methodological Pipeline

1. **Exploratory Data Analysis (EDA)** — descriptive statistics, feature distributions, correlation/covariance matrices, and data quality checks (no missing values detected).
2. **Outlier Handling** — IQR-based removal on the Training Set (6,462 samples remaining out of 7,984 after removing 1,522 anomalous observations). On the Test Set, outliers are instead handled through *clipping* to the bounds calculated from the training data, preserving the integrity of the evaluation set.
3. **Dimensionality Analysis (PCA)** — the first principal component explains only 46.42% of the variance; 10 components are required to explain 95%, indicating that the information is distributed across most electrodes.
4. **Data Leakage Prevention** — all transformations (`StandardScaler`, PCA) are encapsulated in Scikit-learn `Pipeline` objects and fitted independently within each Cross-Validation fold.
5. **Training and Hyperparameter Tuning** — model-specific hyperparameter optimization, including Grid Search for the RBF SVM and Cost-Complexity Pruning for the Decision Tree.
6. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrices, ROC/AUC curves, training time, and statistical validation through Repeated Stratified Cross-Validation (10 folds × 5 repeats).

## Models

| Model                    | Features                      | Notes                                                                    |
| ------------------------ | ----------------------------- | ------------------------------------------------------------------------ |
| **Gaussian Naive Bayes** | 10 principal components (PCA) | Probabilistic benchmark; PCA used to obtain an orthogonal representation |
| **Decision Tree**        | 14 original sensors           | Cost-Complexity Pruning to reduce overfitting; no scaling required       |
| **Linear SVM**           | 14 original sensors           | Evaluated as a linear baseline                                           |
| **RBF SVM**              | 14 original sensors           | Optimized through Cross-Validation over `C ∈ {0.1, 1, 10, 100}`          |

## Results

Metrics calculated on the isolated Test Set (weighted average for Precision/Recall/F1):

| Model                        | Accuracy   | Precision | Recall   | F1-Score | AUC        |
| ---------------------------- | ---------- | --------- | -------- | -------- | ---------- |
| Majority Class Baseline      | 0.5348     | —         | —        | —        | —          |
| Gaussian NB (PCA)            | 0.6320     | 0.63      | 0.63     | 0.63     | 0.6763     |
| Linear SVM                   | 0.6396     | 0.64      | 0.64     | 0.64     | 0.6640     |
| Decision Tree (Post-Pruning) | 0.7308     | 0.73      | 0.73     | 0.73     | 0.7539     |
| **RBF SVM**                  | **0.7700** | **0.77**  | **0.77** | **0.77** | **0.8206** |

### Training Time

Training times include the complete pipeline for Naive Bayes and SVM models.

| Model             | Time (s) |
| ----------------- | -------: |
| Gaussian NB (PCA) |   0.0103 |
| Decision Tree     |   0.1071 |
| RBF SVM           |   6.8101 |
| Linear SVM        |  12.9371 |

The **RBF SVM** achieved the best overall predictive performance, while the **Decision Tree** provided the best trade-off between accuracy, training speed, and interpretability through explicit decision thresholds expressed in µV.

An important finding was the discrepancy between Repeated Stratified Cross-Validation and the isolated Test Set (e.g., RBF SVM: **96.52% CV accuracy vs. 77.00% Test Set accuracy**). The report discusses this discrepancy as a potential consequence of **temporal data leakage** caused by the strong autocorrelation of high-frequency EEG recordings.

## How to Run

```bash
git clone https://github.com/serenantonini/eeg-eye-state-classification.git
cd eeg-eye-state-classification
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
jupyter notebook eeg_eye_state_analysis.ipynb
```

The notebook automatically downloads the train/test datasets from a public GitHub repository at the beginning of the execution, so no manual dataset download is required.

For the complete methodological analysis, including assumptions, experimental design, results discussion, and critical considerations, see [`report.pdf`](./report.pdf).

## Critical Discussion & Future Work

The report discusses the trade-off between Precision and Recall in different application contexts (e.g., road safety vs. BCI) and proposes two main directions for future development:

* **Sequential analysis** — replacing the point-by-point approach with models capable of exploiting temporal windows, given the strong autocorrelation of EEG signals.
* **Frequency-domain filtering** — extracting Alpha-wave power (8–13 Hz) from the occipital region, which is known to be associated with eye closure, instead of relying exclusively on raw time-domain signal values.
