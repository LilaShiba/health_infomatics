import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class ImageLoad:
    def __init__(self, dataset_path: str, resize_to=(64, 64)):
        """
        Initializes the ImageLoad class.

        Args:
            dataset_path (str): Path to the dataset directory containing images.
            resize_to (tuple): Dimensions to resize images to (default: (64, 64)).
        """
        self.radius = 10
        self.dataset_path = dataset_path
        self.image_tensors = defaultdict(list)
        self.output_dir = 'data/output'
        os.makedirs(self.output_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor()
        ])

        # Load image paths
        self.image_paths = self._load_image_paths()
        self.final_image_tensors, self.all_together_tensors = self.main_loop()

    def main_loop(self) -> dict:
        """
        Processes all images and stores the results.
        """
        all_together_tensors = []
        print('Start batch process...')

        for idx, (fname, img_path, category) in enumerate(tqdm(self.image_paths, desc="Processing images")):
            processed_images = [
                self._open_img(img_path),
                self._add_gaussian_blurr(img_path),
                self._rotate_180(img_path),
                self._rotate_90_clockwise(img_path),
                self._rotate_90_counter_clockwise(img_path)
            ]

            for idx_i, img in enumerate(processed_images):
                id_ = f"{idx + 1}_{idx_i}"
                self.save_to_folders(fname, img, id_, category)

        print('Batch process completed.')
        return (self.image_tensors, all_together_tensors)

    def _open_img(self, image_path: str) -> np.ndarray:
        """Opens and transforms the image into NumPy array form."""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img)
        # Convert to HWC format (Height, Width, Channels)
        return img_tensor.permute(1, 2, 0).numpy()

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

    def _bw(self, image_path: str, blur: bool = False) -> np.ndarray:
        """Converts the image to black and white, with optional blurring."""
        image = Image.open(image_path).convert('L')
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(self.radius))
        img_tensor = self.transform(image)
        return img_tensor.squeeze().numpy()  # Convert to 2D array for grayscale

    def _rotate_90_counter_clockwise(self, image_path: str) -> np.ndarray:
        """Rotates the image 90 degrees counterclockwise."""
        return self._rotate_image(image_path, flip_code=0)

    def _rotate_90_clockwise(self, image_path: str) -> np.ndarray:
        """Rotates the image 90 degrees clockwise."""
        return self._rotate_image(image_path, flip_code=1)

    def _rotate_180(self, image_path: str) -> np.ndarray:
        """Rotates the image 180 degrees."""
        return self._rotate_image(image_path, flip_code=-1)

    def _rotate_image(self, image_path: str, flip_code: int) -> np.ndarray:
        """Helper function to read and rotate an image."""
        image = cv2.imread(image_path)
        return cv2.flip(image, flip_code)

    def _add_gaussian_blurr(self, image_path: str) -> np.ndarray:
        """Applies Gaussian blur to the image."""
        image = cv2.imread(image_path)
        return cv2.GaussianBlur(image, (7, 7), 0)

    def _add_gaussian_noise(self, image_path: str, mu: float = 0, sigma: float = 25) -> np.ndarray:
        """Adds Gaussian noise to the image."""
        image = cv2.imread(image_path)
        noise = np.random.normal(mu, sigma, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

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
