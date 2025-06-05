import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

class ImgPreprocessing:
    def __init__(self, image):
        self.image = image

    def Preprocessing(self):
        # Resize image to 90x90 and adjust contrast/brightness
        self.image = cv2.resize(self.image, (90, 90), interpolation=cv2.INTER_CUBIC)
        self.image = cv2.convertScaleAbs(self.image, alpha=1.35, beta=45)

    def FeatureExtraction(self):
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(self.image)
        # Create a 3x3 summary matrix with the mean value of each block
        resumen = np.zeros((3, 3), dtype=float)
        step = self.image.shape[0] // 3  # Assumes square image
        for i in range(3):
            for j in range(3):
                block = self.image[(i * step):(i + 1) * step, (j * step):(j + 1) * step]
                mean_value = np.mean(block)
                resumen[i, j] = mean_value
        return resumen, np.sqrt((minLoc[0] - maxLoc[0]) ** 2 + (minLoc[1] - maxLoc[1]) ** 2)

Imgs_route = '.\data/raw' 
Out_route_90 = '.\data/processed/90x90'
Out_route_3 = '.\data/processed/3x3'
os.makedirs(Out_route_90, exist_ok=True)
os.makedirs(Out_route_3, exist_ok=True)

# Search for images in the raw directory
imgs = sorted(glob(os.path.join(Imgs_route, 'FSS_0632.8nm_00.3mm_*.tiff')))
print(f"{len(imgs)} images were found.")

# Calculate temperatures and categories
temps = [round(x * 0.1, 1) for x in range(len(imgs))]
categories = [int(t // 10) for t in temps]  # 0 to 9

Intensities = []
Int_9pix = []
pix_distances = []

for idx, route in enumerate(imgs):
    img = cv2.imread(route, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image could not be read: {route}")
        continue

    # Initialize and process the image
    ProcessedImg = ImgPreprocessing(img)
    ProcessedImg.Preprocessing()
    processed_90x90 = ProcessedImg.image  # 90x90 processed image
    Intensities.append(processed_90x90.flatten())

    # Feature extraction to get a 3x3 matrix
    resumen_3x3, gradient = ProcessedImg.FeatureExtraction()
    # Convert to uint8 so it can be saved as an image
    resumen_3x3_uint8 = np.clip(resumen_3x3, 0, 255).astype(np.uint8)
    Int_9pix.append(np.append(resumen_3x3_uint8.flatten(), gradient))

    temp = temps[idx]
    out_name = f"Speckle {round(temp, 1)}C.tiff"
    out_path_90 = os.path.join(Out_route_90, out_name)
    out_path_3 = os.path.join(Out_route_3, out_name)

    # Save both processed images
    cv2.imwrite(out_path_90, processed_90x90)
    cv2.imwrite(out_path_3, resumen_3x3_uint8)
    




# DataFrame for 90x90 intensities
intensity_df = pd.DataFrame(Intensities)
intensity_df['temperature'] = temps
intensity_df['category'] = categories

# DataFrame for 3x3 features + gradient
features_df = pd.DataFrame(Int_9pix)
features_df['temperature'] = temps
features_df['category'] = categories

# Save both DataFrames as CSV
intensity_df.to_csv('.\data/processed/intensity_90x90.csv', index=False)
features_df.to_csv('.\data/processed/features.csv', index=False)

print("Saved: intensity_90x90.csv and features_3x3_gradient.csv")
