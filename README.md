# Fiber Specklegram Analysis Pipeline

---

## 📋 Project Overview

This project implements a complete pipeline for **speckle pattern analysis** from multimode fiber sensors. The workflow covers **data preprocessing, feature extraction, statistical analysis, machine learning classification, and results visualization**. The goal is to enable robust temperature or event detection from raw specklegram images using reproducible code and interpretable metrics.

---

## 🗂️ Directory and File Structure

├── data/
│ └── raw/ # Raw .tiff images (input specklegrams)
│ └── processed/
│ ├── 90x90/ # Processed 90x90 images
│ ├── 3x3/ # Feature images (3x3 blocks)
│ ├── intensity_90x90.csv # Intensity dataframe
│ └── features.csv # Feature dataframe (mean blocks + gradient + category)
├── results/
│ ├── tables/ # Output CSVs for statistics, metrics, and summaries
│ └── figures/
│ ├── kde/ # KDE plots per feature
│ └── box/ # Boxplots per feature
│ └── kde_grid.png # Grid of KDEs
│ └── box_grid.png # Grid of boxplots
├── src/
│ ├── preprocessing.py # Image and feature preprocessing
│ ├── analysis.py # Statistical analysis, ML, and metrics
│ ├── visualization.py # Automated plots for features and metrics
│ └── notebooks/ # Jupyter notebooks for end-to-end demonstration
└── README.md # This file


---

## ⚙️ Main Processing Logic

### 1. **Data Preprocessing**
- **Input:** `.tiff` specklegram images (`/data/raw/`)
- **Output:**  
  - Rescaled 90x90 images (`/data/processed/90x90/`)
  - 3x3 block-mean images (`/data/processed/3x3/`)
  - CSVs:  
    - `intensity_90x90.csv`: All 90x90 pixel values per sample  
    - `features.csv`: 9 block means, gradient, temperature, category

### 2. **Feature Engineering**
- **Block averaging**: Each image divided into 3x3 blocks, mean value computed per block.
- **Gradient feature**: Distance between minimum and maximum intensity pixels.
- **Temperature/category assignment**: Based on acquisition order or filename.

### 3. **Statistical Analysis**
- **Normality test:** Shapiro-Wilk per feature/category
- **Kruskal-Wallis test:** Group comparison for each feature
- **Fisher’s discriminant ratio:** Separability between classes
- **AUC scores:** ROC AUC for each feature, one-vs-all

### 4. **Machine Learning**
- **Random Forest classifier**: Predicts category from features, with train/test split
- **Metrics:** Accuracy, precision, recall, F1, confusion matrix, and per-class report

### 5. **Visualization**
- **KDE plots:** Per feature and per category, plus summary grid
- **Boxplots:** Per feature and per category, plus summary grid
- **Sample images:** Example 90x90 and 3x3 processed images

---

## 📁 Main Results

- **/results/tables/**: All metrics, statistical test results, and summaries as CSV files (e.g., `classification_metrics.csv`, `feature_means_by_category.csv`, etc.)
- **/results/figures/kde/**: Individual KDE plots (`kde_Feature_1.png`, ...)
- **/results/figures/box/**: Individual boxplots (`box_Feature_1.png`, ...)
- **/results/figures/kde_grid.png, box_grid.png**: All features in one grid image
- **/data/processed/features.csv**: Main feature table for modeling and stats

---

## 🚀 How to Run

1. **Prepare Data:**  
   Place raw `.tiff` images in `data/raw/`.

2. **Run Preprocessing:**  
   Use `src/preprocessing.py` to generate processed images and feature CSVs.

3. **Run Analysis and Metrics:**  
   Use `src/analysis.py` to compute statistics, ML metrics, and generate results in `/results/tables/`.

4. **Run Visualization:**  
   Use `src/visualization.py` or provided Jupyter notebooks to produce plots (saved in `/results/figures/`).

5. **Jupyter Notebooks:**  
   For step-by-step demonstrations, open and run any notebook in `src/notebooks/`.

---

## 🛠️ Methods Summary

- **Image normalization:** OpenCV, blockwise mean, and contrast/brightness adjustment.
- **Feature engineering:** Numpy operations, pandas DataFrame creation.
- **Statistical testing:** scipy.stats (shapiro, kruskal), custom scatter matrix code.
- **ML modeling:** sklearn (RandomForestClassifier, train_test_split, metrics).
- **Visualization:** seaborn, matplotlib.
- **File I/O:** All results and figures are written as CSV or PNG for reproducibility.

---

## 🔎 Example Results Preview

- **Classification accuracy (RF):**  
  ![](results/tables/classification_metrics.csv)

- **KDE plot example:**  
  ![](results/figures/kde/kde_Feature_1.png)

- **Boxplot grid:**  
  ![](results/figures/box_grid.png)

---

## 📚 Further Notes

- All steps are modular and well-commented.
- If you add new features or images, re-run preprocessing and downstream scripts.
- Each CSV and plot is timestamped or reproducibly generated.
- Works with Python 3.x, requires `numpy`, `pandas`, `opencv-python`, `matplotlib`, `seaborn`, `scikit-learn`, and `scipy`.

---

## 👨‍💻 Authors

- Isaac Huertas-Montes – fiber speckle analysis pipeline  
- For academic use and reproducible research.

---

