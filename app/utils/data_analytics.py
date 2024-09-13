import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar


class ImageDatasetAnalyzer:
    def __init__(self, dataset_path):
        """
        Initialize with the path to the dataset.
        """
        self.dataset_path = dataset_path
        self.image_paths = self._load_image_paths()
        self.image_data = []

    def _load_image_paths(self):
        """
        Loads image paths from the dataset folder, handling .tiff, .jpg, .jpeg, and .png images.
        """
        image_extensions = ['.tiff', '.tif', '.jpg',
                            '.jpeg', '.png']  # Supported image extensions
        image_paths = [
            os.path.join(self.dataset_path, fname)
            for fname in os.listdir(self.dataset_path)
            if any(fname.lower().endswith(ext) for ext in image_extensions)
        ]
        return image_paths

    def load_images(self, resize_to=(64, 64)):
        """
        Load images, resize, and normalize them.
        """
        transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor()
        ])

        for img_path in tqdm(self.image_paths, desc="Analyzing images"):
            try:
                # Convert all images to RGB for consistency
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                self.image_data.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    def analyze_color_distribution(self):
        """
        Analyze color distribution across the dataset and compute the mean values for Red, Green, Blue,
        and Brightness for each image. Store them in a pandas DataFrame with 4 columns, with a progress bar.
        """
        color_stats = {
            'Red': [],
            'Green': [],
            'Blue': [],
            'Brightness': []
        }

        # Use tqdm to add a progress bar for analyzing images
        for image_tensor in tqdm(self.image_data, desc="Analyzing images"):
            image_np = image_tensor.numpy()

            # Calculate mean for each color channel
            red_mean = np.mean(image_np[0])
            green_mean = np.mean(image_np[1])
            blue_mean = np.mean(image_np[2])

            # Calculate brightness as the average across all color channels
            brightness = np.mean([red_mean, green_mean, blue_mean])

            # Append the calculated values to the color_stats dictionary
            color_stats['Red'].append(red_mean)
            color_stats['Green'].append(green_mean)
            color_stats['Blue'].append(blue_mean)
            color_stats['Brightness'].append(brightness)

        # Convert the color stats to a pandas DataFrame
        df = pd.DataFrame(color_stats)

        # Display the DataFrame
        print(df)

        # Optionally, you can plot the distributions if needed
        df.plot(kind='box', title='Color Channel and Brightness Distribution')
        plt.show()


# Usage Example
if __name__ == "__main__":
    # Replace with the actual path to your image dataset
    dataset_path = 'app/Blood_Cancer_TIFFS'
    print("finding images")
    analyzer = ImageDatasetAnalyzer(dataset_path)
    print("load images")
    analyzer.load_images()
    print("analyzing images")
    analyzer.analyze_color_distribution()
