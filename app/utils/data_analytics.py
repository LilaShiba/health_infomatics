import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from load_images import ImageLoad  # Assuming ImageLoad is saved in load_images.py


class ImageDatasetAnalyzer(ImageLoad):
    """
    Processes a dataset folder and generates:
    - image_dict: A dictionary containing the original images and their transformations (blurred, rotated, etc.) in RGB and Grayscale.
    - df: A pandas DataFrame with RGB & Grayscale averages per image.
    """

    def __init__(self, dataset_path: str, resize_to=(64, 64)):
        """
        Initialize the dataset analyzer with the dataset path and transformations.
        """
        # Inherit everything from ImageLoad
        super().__init__(dataset_path=dataset_path, resize_to=resize_to)

    def tensors_to_df(self) -> dict:
        """
        Converts each key in the final_image_tensors dictionary into a Pandas DataFrame.

        The DataFrame contains statistics (mean, std) for each tensor in the list under a given key.

        Args:
                final_image_tensors (dict): A dictionary where keys are transformation names and values are lists of tensors.

        Returns:
                dict: A dictionary where keys are transformation names and values are DataFrames with image statistics.
            """
        df_dict = {}

        for transform_name, tensor_list in self.final_image_tensors.items():
            # Initialize lists to store statistics for each tensor
            stats_list = []

            for tensor in tensor_list:
                if isinstance(tensor, torch.Tensor):
                    # Convert tensor to numpy for easy manipulation
                    np_tensor = tensor.numpy()

                    # Check if it's an RGB image (with 3 channels) or grayscale (single channel)
                    if np_tensor.shape[0] == 3:  # RGB
                        red_mean = np.mean(np_tensor[0])
                        green_mean = np.mean(np_tensor[1])
                        blue_mean = np.mean(np_tensor[2])
                        red_std = np.std(np_tensor[0])
                        green_std = np.std(np_tensor[1])
                        blue_std = np.std(np_tensor[2])
                        stats_list.append({
                            'Red Mean': red_mean,
                            'Green Mean': green_mean,
                            'Blue Mean': blue_mean,
                            'Red Std': red_std,
                            'Green Std': green_std,
                            'Blue Std': blue_std
                        })
                    else:  # Grayscale
                        gray_mean = np.mean(np_tensor)
                        gray_std = np.std(np_tensor)
                        stats_list.append({
                            'Gray Mean': gray_mean,
                            'Gray Std': gray_std
                        })

            # Create a DataFrame for this transformation
            df = pd.DataFrame(stats_list)
            df_dict[transform_name] = df

        return df_dict

    def batch_process_folders(self) -> dict:
        """
        Batch processes all folders in the dataset path and returns image tensors.
        """
        print('Batch processing folders...')
        return self.main_loop()  # Use inherited method from ImageLoad


# Usage Example
if __name__ == "__main__":
    print('Initializing tool')
    analyzer = ImageDatasetAnalyzer(
        '/Users/kjams/Desktop/research/health_informatics/app/data/testing_data')
    print('Start batch process...')
    image_dict = analyzer.batch_process_folders()
    print('Batch process completed.')
    # print(image_dict['blur'])  # Example access to processed images
    dfs = analyzer.tensors_to_df()
    # Check dfs
    for key, item in dfs.items():
        print(key)
        print(item.head(5))
