import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
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

    def load_images(self, resize_to=(64, 64), save_blurred=True) -> dict:
        """
        Load images, resize, and normalize them.

        Parameters:
        - resize_to: tuple, resize dimensions for the images.
        - save_blurred: bool, whether to save blurred images to output directory.

        Return:
        res = {
            'rgb': [],
            'bw': [],
            'rgb_blurr': [],
            'bw_blurr': []
        }

        """
        transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor()
        ])

        output_dir = 'data/output'
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        res = {
            'rgb': [],
            'bw': [],
            'rgb_blurr': [],
            'bw_blurr': []
        }

        for img_path in tqdm(self.image_paths, desc="Gathering images"):
            try:
                # Convert all images to RGB for consistency
                image = Image.open(img_path).convert('RGB')

                # Make a copy in Grey Scale
                bw_image = image.convert('L')

                # Apply Gaussian Blur
                blurred_image = image.filter(
                    ImageFilter.GaussianBlur(radius=3))

                blurred_image_bw = bw_image.filter(
                    ImageFilter.GaussianBlur(radius=3))

                # Save blurred image if required
                if save_blurred:
                    # Generate a unique filename based on original image name
                    base_name = os.path.basename(img_path)
                    blurred_image_name = os.path.splitext(
                        base_name)[0] + "_blurred.jpg"
                    blurred_image_path = os.path.join(
                        output_dir, blurred_image_name)
                    blurred_image.save(blurred_image_path)

                # Transform to tensor and store
                image_tensor = transform(image)
                image_tensor_bw = transform(bw_image)
                blurred_image_tensor = transform(blurred_image)
                blurred_image_bw_tensor = transform(blurred_image_bw)

                self.image_data.append(image_tensor)

                res['rgb'].append(image_tensor)
                res['bw'].append(image_tensor_bw)
                res['rgb_blurr'].append(blurred_image_tensor)
                res['bw_blurr'].append(blurred_image_bw_tensor)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

            return res

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
            'Brightness': [],
        }

        # Use tqdm to add a progress bar for analyzing images
        for image_tensor in tqdm(self.image_data, desc="Analyzing images"):
            try:
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
            except Exception as e:
                print(f"Error analyzing image: {e}")

        # Convert the color stats to a pandas DataFrame
        df = pd.DataFrame(color_stats)

        # Display the DataFrame (optional, just showing the first few rows)
        print(df.head())

        # Plot the distributions
        plt.figure(figsize=(10, 5))
        df.plot(
            kind='box', title='Color Channel and Brightness Distribution', ax=plt.gca())

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
        output_dir = "data/output"
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over all subfolders in the dataset_path
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)

            # Ensure it's a directory
            if os.path.isdir(folder_path):
                print(f"Processing dataset in folder: {folder}")
                temp_obj = ImageDatasetAnalyzer(folder_path)
                _ = temp_obj.load_images()

                # Save plot for each dataset with a unique filename
                output_file = os.path.join(
                    output_dir, f"{folder}_color_distribution.png")
                # Analyze Images
                # TODO: Take in DICT
                temp_obj.analyze_color_distribution(output_file=output_file)
                # TODO: return Dataframe after processing


# Usage Example
if __name__ == "__main__":
    # Base path to your dataset
    dataset_path = 'app/data/stages'
    dataset_path = "data/testing_data"
    print('initalizing tool')
    analyzer = ImageDatasetAnalyzer(dataset_path)
    print('Start batch process...')
    analyzer.batch_process_folders()
    print('Batch process completed.')
