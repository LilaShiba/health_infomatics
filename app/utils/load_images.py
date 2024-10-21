import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
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
        # blurr radius
        self.radius = 3
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
        Iterates over all image paths, processing each one and storing the results.
        """
        all_together_tensors = []
        print('Start batch process...')
        for img_path in tqdm(self.image_paths, desc="Processing images"):
            original = self._open_img(img_path)
            blurred_tensor = self._blur(img_path)
            bw_tensor = self._bw(img_path)
            bw_blurr = self._bw(img_path, True)
            rotate_90_tensor, rotate_180_tensor = self._rotate(img_path)
            # update tensor dict for later analysis
            self.image_tensors['org'].append(original)
            self.image_tensors['blur'].append(blurred_tensor)
            self.image_tensors['bw'].append(bw_tensor)
            self.image_tensors['bw_blurr'].append(bw_blurr)
            self.image_tensors['img_90'].append(rotate_90_tensor)
            self.image_tensors['img_180'].append(rotate_180_tensor)
            # For Neural Network : )
            all_together_tensors.append(original)
            all_together_tensors.append(blurred_tensor)
            all_together_tensors.append(rotate_90_tensor)
            all_together_tensors.append(rotate_180_tensor)

        print('Batch process completed.')
        return (self.image_tensors, all_together_tensors)

    def _open_img(self, image_path):
        '''
        returns original image in tensor form
        '''
        img = Image.open(image_path).convert('RGB')
        return self.transform(img)

    def _blur(self, image_path: str, radius: int = None) -> torch.Tensor:
        """
        Applies a Gaussian blur to the image at the given path.
        """
        if not radius:
            radius = self.radius
        image = Image.open(image_path).convert('RGB')
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius))
        return self.transform(blurred_image)

    def _bw(self, image_path: str, blur=False) -> torch.Tensor:
        """
        Converts the image at the given path to black and white.
        """
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        if not blur:
            return self.transform(image)
        else:
            return self.transform(image.filter(ImageFilter.GaussianBlur(self.radius)))

    def _rotate(self, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates the image at the given path by 90 and 180 degrees.
        """
        image = Image.open(image_path).convert('RGB')
        rotate_90_image = image.rotate(90)
        rotate_180_image = image.rotate(180)
        return self.transform(rotate_90_image), self.transform(rotate_180_image)

    def _load_image_paths(self) -> list[str]:
        """
        Recursively loads image file paths from the dataset directory and its subdirectories with supported image formats.

        Returns:
            list[str]: List of image file paths.
        """
        image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
        image_paths = []

        # Walk through all subdirectories and collect image paths
        for root, _, files in os.walk(self.dataset_path):
            for fname in files:
                if any(fname.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, fname))

        return image_paths
