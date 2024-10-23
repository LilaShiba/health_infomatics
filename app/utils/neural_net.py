import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from load_images import ImageLoad  # Assuming ImageLoad is saved in load_images.py


# make directories
# classifications based on folders
# training
# test

# add filters

# gaussian
# median
# blurr

# add noise
# Gaussian
# sp


if __name__ == "__main__":
    folder_path = "/Users/kjams/Desktop/research/health_informatics/app/data/testing_data"
    print('init')
    images = ImageLoad(folder_path)
    dict_of_image_tensors, image_df = images.main_loop()
    print(dict_of_image_tensors)
