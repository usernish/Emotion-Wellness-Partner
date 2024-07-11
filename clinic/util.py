# utils.py

import pandas as pd
from PIL import Image
import numpy as np

def load_fer2013_dataset(csv_file_path):
    df = pd.read_csv(csv_file_path)
    images = df['pixels'].values
    labels = df['emotion'].values
    return images, labels

def preprocess_images(image_data, size=(48, 48)):
    images = []
    for pixels in image_data:
        pixel_array = np.array(pixels.split(), dtype='uint8')
        image = pixel_array.reshape(size)
        image = Image.fromarray(image).convert('L')
        image = image.resize(size)
        image = np.array(image) / 255.0
        images.append(image)
    return np.array(images)
