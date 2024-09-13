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

        for img_path in tqdm(self.image_paths, desc="Gathering images"):
            try:
                # Convert all images to RGB for consistency
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                self.image_data.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    def analyze_color_distribution(self, output_file=None):
        """
        Analyze color distribution across the dataset and compute the mean values for Red, Green, Blue,
        and Brightness for each image. Store them in a pandas DataFrame with 4 columns, with a progress bar.

        :param output_file: Optional, specify the file path to save the plot.
        :return: DataFrame with color stats for further analysis.
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

        # Display the DataFrame (optional, just showing the first few rows)
        print(df.head())

        # Plot the distributions
        df.plot(
            kind='box', title=f'{output_file} Color Channel and Brightness Distribution')

        # Save plot if output_file is provided
        if output_file:
            plt.savefig(output_file)
        plt.show()

        return df  # Return DataFrame for further use if needed

    def batch_process_folders(self):
        """
        Iterate over the base dataset folder containing subfolders, each of which is a dataset to be processed.
        Each dataset will be processed and the results saved in the 'data/output' folder.
        """
        output_dir = "app/data/output"
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over all subfolders in the dataset_path
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)

            # Ensure it's a directory
            if os.path.isdir(folder_path):
                print(f"Processing dataset in folder: {folder}")
                temp_obj = ImageDatasetAnalyzer(folder_path)
                temp_obj.load_images()

                # Save plot for each dataset with a unique filename
                output_file = os.path.join(
                    output_dir, f"{folder}_color_distribution.png")
                temp_obj.analyze_color_distribution(output_file=output_file)


# Usage Example
if __name__ == "__main__":
    # Base path to your dataset
    dataset_path = 'app/data/stages'
    analyzer = ImageDatasetAnalyzer(dataset_path)
    analyzer.batch_process_folders()
