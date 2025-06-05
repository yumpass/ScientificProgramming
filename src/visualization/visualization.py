import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

# Set paths
result_dir = r'.\results\tables'
kde_dir = r'.\results\figures\kde'
box_dir = r'.\results\figures\box'
features_path = os.path.join(r'.\data\processed', 'features.csv')

print("==== 1. Loading and displaying result tables (CSV) ====")
csv_files = [
    'normality_shapiro_results.csv',
    'kruskal_wallis_results.csv',
    'feature_means_by_category.csv',
    'overall_mean_by_feature.csv',
    'fisher_discriminant_ratios.csv',
    'auc_scores_by_feature_class.csv',
    'samples_per_category_before_filter.csv',
    'samples_per_category_after_filter.csv',
    'classification_metrics.csv',
    'classification_report.csv'
]

for csv in csv_files:
    csv_path = os.path.join(result_dir, csv)
    if os.path.exists(csv_path):
        print(f"\n----- {csv} -----")
        df = pd.read_csv(csv_path)
        print(df.head())
    else:
        print(f"{csv} not found.")

print("\n==== 2. Loading features for sample visualization ====")
if os.path.exists(features_path):
    Features = pd.read_csv(features_path)
    print("Features loaded.")
    print(Features.head())
    sample_idx = 0
    # Visualize first sample as a bar plot
    feature_cols = [col for col in Features.columns if col not in ['temperature', 'category', 'Categories']]
    print("\nSample feature values (row 0):")
    print(Features.loc[sample_idx, feature_cols])
    plt.figure(figsize=(12, 4))
    plt.bar(feature_cols, Features.loc[sample_idx, feature_cols])
    plt.title("Feature values for first sample")
    plt.ylabel("Value")
    plt.xlabel("Feature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("features.csv not found.")

print("\n==== 3. Visualizing KDE plots (first 3 as example) ====")
kde_files = [f for f in os.listdir(kde_dir) if f.endswith('.png') and f.startswith('kde_')]
kde_files = sorted(kde_files)
for file in kde_files[:3]:
    img_path = os.path.join(kde_dir, file)
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(file)
    plt.show()

print("\n==== 4. Visualizing Boxplots (first 3 as example) ====")
box_files = [f for f in os.listdir(box_dir) if f.endswith('.png') and f.startswith('box_')]
box_files = sorted(box_files)
for file in box_files[:3]:
    img_path = os.path.join(box_dir, file)
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(file)
    plt.show()

print("\n==== 5. Visualizing KDE and Boxplot grids ====")
for grid_file, grid_dir in [('kde_grid.png', kde_dir), ('box_grid.png', box_dir)]:
    grid_path = os.path.join(grid_dir, grid_file)
    if os.path.exists(grid_path):
        img = Image.open(grid_path)
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(grid_file)
        plt.show()
    else:
        print(f"{grid_file} not found.")

print("\n==== 6. Example: Visualizing raw and processed images ====")
# Show a sample processed image (90x90) and 3x3 image if available
processed_img_dir = r'.\data\processed\90x90'
processed_img_files = sorted([f for f in os.listdir(processed_img_dir) if f.endswith('.tiff')])
if processed_img_files:
    img_path = os.path.join(processed_img_dir, processed_img_files[0])
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Example processed 90x90 image")
    plt.axis('off')
    plt.show()
else:
    print("No 90x90 images found.")

processed_3x3_dir = r'.\data\processed\3x3'
processed_3x3_files = sorted([f for f in os.listdir(processed_3x3_dir) if f.endswith('.tiff')])
if processed_3x3_files:
    img_path = os.path.join(processed_3x3_dir, processed_3x3_files[0])
    img = Image.open(img_path)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title("Example processed 3x3 image")
    plt.axis('off')
    plt.show()
else:
    print("No 3x3 images found.")

print("\n==== Visualization summary complete. ====")
