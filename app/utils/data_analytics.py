import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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
        Loads image paths from the dataset folder.
        """
        image_extensions = ['.jpg', '.jpeg', '.png']
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
        
        for img_path in self.image_paths:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image)
            self.image_data.append(image_tensor)

    def analyze_color_distribution(self):
        """
        Analyze color distribution across the dataset.
        """
        color_stats = {'red': [], 'green': [], 'blue': []}
        
        for image_tensor in self.image_data:
            image_np = image_tensor.numpy()
            color_stats['red'].append(np.mean(image_np[0]))
            color_stats['green'].append(np.mean(image_np[1]))
            color_stats['blue'].append(np.mean(image_np[2]))
        
        # Plot the color distribution
        df = pd.DataFrame(color_stats)
        df.plot(kind='box', title='Color Channel Distribution')
        plt.show()

    def calculate_pixel_intensity_variance(self):
        """
        Calculate pixel intensity variance across all images to measure diversity.
        """
        variances = []
        
        for image_tensor in self.image_data:
            image_np = image_tensor.numpy()
            pixel_intensity = np.mean(image_np, axis=0)  # Averaging across color channels
            variances.append(np.var(pixel_intensity))
        
        # Return the mean variance across all images
        mean_variance = np.mean(variances)
        return mean_variance

    def diversity_score(self):
        """
        Calculate a diversity score based on pixel intensity variance.
        """
        variance = self.calculate_pixel_intensity_variance()
        # Normalized diversity score (arbitrary scale)
        score = variance / np.max(variance)
        return score

# Usage Example
if __name__ == "__main__":
    dataset_path = './path_to_your_images'
    analyzer = ImageDatasetAnalyzer(dataset_path)
    
    analyzer.load_images()
    analyzer.analyze_color_distribution()
    
    pixel_variance = analyzer.calculate_pixel_intensity_variance()
    print(f'Pixel Intensity Variance: {pixel_variance}')
    
    diversity_score = analyzer.diversity_score()
    print(f'Diversity Score: {diversity_score}')
