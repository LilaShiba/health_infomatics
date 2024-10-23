import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class ImageLoad:
    def __init__(self, dataset_path: str, resize_to=(64, 64)):
        """
        Initializes the ImageLoad class.

        Args:
            dataset_path (str): Path to the dataset directory containing images.
            resize_to (tuple): Dimensions to resize images to (default: (64, 64)).
        """
        self.resize_to = resize_to
        self.dataset_path = dataset_path
        self.image_tensors = defaultdict(list)
        self.output_dir = 'data/output'
        os.makedirs(self.output_dir, exist_ok=True)

        # Load image paths
        self.image_paths = self._load_image_paths()
        self.final_image_arrays = self.main_loop()

    def main_loop(self) -> dict:
        """
        Processes all images and stores the results.
        """
        all_together_arrays = []
        print('Start batch process...')

        for idx, (fname, img_path, category) in enumerate(tqdm(self.image_paths, desc="Processing images")):
            image = self._open_img(img_path)

            processed_images = [
                image,
                self._add_gaussian_blurr(image),
                self._rotate_180(image),
                self._rotate_90_clockwise(image),
                self._rotate_90_counter_clockwise(image)
            ]

            for idx_i, img in enumerate(processed_images):
                id_ = f"{idx + 1}_{idx_i}"
                self.save_to_folders(fname, img, id_, category)

        print('Batch process completed.')
        return all_together_arrays

    def _open_img(self, image_path: str) -> np.ndarray:
        """Opens and transforms the image into NumPy array form."""
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, self.resize_to)  # Resize the image
        # TODO add noise to each image
        return img_resized

    def save_to_folders(self, fname: str, image: np.ndarray, idx: str, category: str):
        """Saves the processed image to the specified category folder."""
        # Ensure image is in the correct format
        if not isinstance(image, np.ndarray):
            raise ValueError("The image must be a NumPy array.")

        # Create the category folder if it doesn't exist
        category_path = os.path.join(self.output_dir, 'processed', category)
        os.makedirs(category_path, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(category_path, f"{idx}_{fname}")
        print(f"Saving image to: {file_path}")

        # Save the image
        success = cv2.imwrite(file_path, image)
        if not success:
            raise IOError(f"Failed to save the image at {file_path}")

    def _rotate_90_counter_clockwise(self, image: np.ndarray) -> np.ndarray:
        """Rotates the image 90 degrees counterclockwise."""
        return np.rot90(image, k=1)

    def _rotate_90_clockwise(self, image: np.ndarray) -> np.ndarray:
        """Rotates the image 90 degrees clockwise."""
        return np.rot90(image, k=-1)

    def _rotate_180(self, image: np.ndarray) -> np.ndarray:
        """Rotates the image 180 degrees."""
        return cv2.flip(image, -1)

    def _add_gaussian_blurr(self, image: np.ndarray) -> np.ndarray:
        """Applies Gaussian blur to the image."""
        return cv2.GaussianBlur(image, (7, 7), 0)

    def _load_image_paths(self) -> list[tuple[str, str, str]]:
        """
        Recursively loads image file paths from the dataset directory.

        Returns:
            list[tuple[str, str, str]]: List of image file paths with their respective category.
        """
        image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
        image_paths = []

        # Loop through the main dataset directory
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if os.path.isdir(category_path):
                for fname in os.listdir(category_path):
                    if any(fname.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(
                            (fname, os.path.join(category_path, fname), category))

        return image_paths
