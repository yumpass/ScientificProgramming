import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kruskal
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Directory for results and tables
result_dir = r'.\results\tables'
os.makedirs(result_dir, exist_ok=True)

# Read features from CSV
features_path = os.path.join(result_dir, '..', '..', 'data', 'processed', 'features.csv')
features_path = os.path.abspath(features_path)
kde_dir = os.path.join('.', 'results', 'figures', 'kde')
box_dir = os.path.join('.', 'results', 'figures', 'box')
Features = pd.read_csv(features_path)

# Automatically get feature columns (all columns except known labels/meta)
ignore_cols = ['temperature', 'category', 'Categories']
Features_cols = [col for col in Features.columns if col not in ignore_cols]

# Add 'Categories' column if missing
if 'Categories' not in Features.columns:
    if 'category' in Features.columns:
        Features['Categories'] = Features['category']
    else:
        raise Exception("No 'Categories' or 'category' column found.")

# --- Normality test with Shapiro-Wilk ---
NormalTestVisualization = []
for i in range(10):
    shapiroData = []
    for col_name in Features_cols:
        data = Features[Features['Categories'] == i][col_name]
        shapiroTest = shapiro(data)
        shapiroData.append(shapiroTest[1])  # p-value
    NormalTestVisualization.append(shapiroData)

NormalTest = pd.DataFrame(NormalTestVisualization, columns=Features_cols)
NormalTest.to_csv(os.path.join(result_dir, 'normality_shapiro_results.csv'), index=False)

# --- Kruskal-Wallis test ---
kruskal_results = []
for feature in Features_cols:
    groups = [group[feature].values for _, group in Features.groupby('Categories')]
    h_stat, p_value = kruskal(*groups)
    kruskal_results.append({'Feature': feature, 'H-statistic': h_stat, 'p-value': p_value})

kruskal_df = pd.DataFrame(kruskal_results)
kruskal_df.to_csv(os.path.join(result_dir, 'kruskal_wallis_results.csv'), index=False)

# --- Means for each feature by Categories ---
feature_means_by_Categories = Features.groupby('Categories')[Features_cols].mean()
feature_means_by_Categories.to_csv(os.path.join(result_dir, 'feature_means_by_category.csv'))

# --- Overall mean for each feature ---
overall_mean = Features[Features_cols].mean()
overall_mean.to_csv(os.path.join(result_dir, 'overall_mean_by_feature.csv'), header=['mean'])

# --- Within-class scatter matrix ---
within_class_scatter = np.zeros((len(Features_cols), len(Features_cols)))
for cat in sorted(Features['Categories'].unique()):
    Categories_df = Features[Features['Categories'] == cat][Features_cols]
    Categories_means = feature_means_by_Categories.loc[cat, Features_cols]
    centered = Categories_df - Categories_means
    within_class_scatter += np.dot(centered.T, centered)

# --- Between-class scatter matrix ---
between_class_scatter = np.zeros((len(Features_cols), len(Features_cols)))
overall_mean_vec = overall_mean.values
for cat, cat_means in feature_means_by_Categories.iterrows():
    n = len(Features[Features['Categories'] == cat])
    mean_diff = (cat_means.values - overall_mean_vec).reshape(-1, 1)
    between_class_scatter += n * (mean_diff @ mean_diff.T)

# --- Fisher's discriminant ratio for each feature ---
fisher_discriminant_ratio = np.diag(np.dot(np.linalg.pinv(within_class_scatter), between_class_scatter))
fdr_df = pd.DataFrame({'Feature': Features_cols, 'FDR': fisher_discriminant_ratio})
fdr_df.to_csv(os.path.join(result_dir, 'fisher_discriminant_ratios.csv'), index=False)

# --- AUC computation ---
X = Features[Features_cols]
y = Features['Categories'].values
auc_results = []
for class_idx in np.unique(y):
    auc_scores = []
    for feature_idx in range(X.shape[1]):
        feature_values = X.iloc[:, feature_idx].values
        target = (y == class_idx).astype(int)
        auc_score = roc_auc_score(target, feature_values)
        if auc_score < 0.5:
            auc_score = 1 - auc_score
        auc_scores.append(auc_score)
    auc_results.append(auc_scores)
AUC_Data = pd.DataFrame(auc_results, columns=Features_cols)
AUC_Data.to_csv(os.path.join(result_dir, 'auc_scores_by_feature_class.csv'), index=False)

# --- Classification: Random Forest metrics ---

# Check class counts before split
class_counts = Features['Categories'].value_counts()
# Save report of number of samples per category before filtering
class_counts.to_csv(os.path.join(result_dir, 'samples_per_category_before_filter.csv'))

# Filter out categories with fewer than 2 samples
valid_classes = class_counts[class_counts >= 2].index
mask = Features['Categories'].isin(valid_classes)

X = Features.loc[mask, Features_cols]
y = Features.loc[mask, 'Categories'].values

# Save report of number of samples per category after filtering
pd.Series(y).value_counts().to_csv(os.path.join(result_dir, 'samples_per_category_after_filter.csv'))

# Now, only classes with at least 2 samples remain
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

metrics_df = pd.DataFrame({
    'Accuracy': [accuracy_score(y_test, y_pred)],
    'Precision': [precision_score(y_test, y_pred, average='macro')],
    'Recall': [recall_score(y_test, y_pred, average='macro')],
    'F1-score': [f1_score(y_test, y_pred, average='macro')]
})
metrics_df.to_csv(os.path.join(result_dir, 'classification_metrics.csv'), index=False)

# Classification report (by class, as DataFrame)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(result_dir, 'classification_report.csv'))


# Calculate grid size (e.g., 3x4 if 10-12 features, etc.)
n_feats = len(Features_cols)
n_cols = 4
n_rows = math.ceil(n_feats / n_cols)

# KDE plots for each feature
for feature in Features_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=Features, x=feature, hue='Categories', fill=True, alpha=0.5, common_norm=False)
    plt.title(f'Density Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(kde_dir, f'kde_{feature}.png'))
    plt.close()

# Boxplots for each feature
for feature in Features_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Categories', y=feature, data=Features)
    plt.title(f'Boxplot of {feature} by Categories')
    plt.xlabel('Categories')
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(os.path.join(box_dir, f'box_{feature}.png'))
    plt.close()
