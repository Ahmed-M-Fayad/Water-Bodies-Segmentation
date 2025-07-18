"""
Water Segmentation Data Handler

This module provides a comprehensive data handler for water segmentation tasks,
including loading TIFF images and PNG labels, preprocessing, normalization,
and dataset splitting for machine learning workflows.

Requirements:
    - numpy
    - PIL (Pillow)
    - tifffile
    - scikit-learn
    - os
    - re

Author: Ahmed Mohammad Fayad
Date: 2025-07-18
Version: 1.0
"""

import os
import re
import numpy as np
from PIL import Image
import tifffile as tiff
from sklearn.preprocessing import MinMaxScaler


class WaterSegmentationDataHandler:
    """
    Handles data loading, preprocessing, and splitting for water segmentation tasks.
    
    This class provides a complete pipeline for processing water segmentation datasets,
    including loading TIFF input images and PNG segmentation labels, normalizing the
    data, and splitting into training, validation, and test sets.
    
    Attributes:
        data_path (str): Root path to the dataset directory
        images_dir (str): Path to the images subdirectory
        labels_dir (str): Path to the labels subdirectory
        input_images (list): List of loaded input images as numpy arrays
        segmentation_labels (list): List of loaded segmentation labels as numpy arrays
        scaled_input_images (list): List of normalized input images
    
    Example:
        >>> handler = WaterSegmentationDataHandler("/path/to/dataset")
        >>> (train_imgs, val_imgs, test_imgs), (train_labels, val_labels, test_labels) = handler.process_all()
        >>> print(f"Training set size: {len(train_imgs)}")
    """
    
    def __init__(self, data_path):
        """
        Initialize the data handler with dataset path.
        
        Args:
            data_path (str): Path to the root directory containing 'images' and 'labels' subdirectories
        
        Raises:
            Warning: If the specified directories don't exist (handled gracefully in load methods)
        """
        self.data_path = data_path
        self.images_dir = os.path.join(data_path, "images")
        self.labels_dir = os.path.join(data_path, "labels")
        self.input_images = []
        self.segmentation_labels = []
        self.scaled_input_images = []
        
    def natural_key(self, file_name):
        """
        Generate a natural sorting key for filenames containing numbers.
        
        This ensures files are sorted in natural order (e.g., file1.tif, file2.tif, file10.tif)
        instead of lexicographical order (e.g., file1.tif, file10.tif, file2.tif).
        
        Args:
            file_name (str): The filename to generate a sorting key for
            
        Returns:
            list: A list of strings and integers for natural sorting
            
        Example:
            >>> handler = WaterSegmentationDataHandler("/path")
            >>> handler.natural_key("image10.tif")
            ['image', 10, '.tif']
        """
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', file_name)]
    
    def load_tiff_images(self, directory):
        """
        Load TIFF images from the specified directory.
        
        Searches for files with .tif or .tiff extensions and loads them using tifffile.
        Images are loaded in natural sorted order and converted to numpy arrays.
        
        Args:
            directory (str): Path to directory containing TIFF files
            
        Returns:
            list: List of numpy arrays representing the loaded images
            
        Prints:
            - Number of TIFF files found
            - Loading progress for each file
            - Shape information for each loaded image
            - Error messages for files that fail to load
        """
        images = []
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return images
        
        tiff_files = sorted([f for f in os.listdir(directory) 
                            if f.endswith(('.tif', '.tiff'))], key=self.natural_key)
        
        print(f"Found {len(tiff_files)} TIFF files")
        
        for filename in tiff_files:
            file_path = os.path.join(directory, filename)
            try:
                image = np.array(tiff.imread(file_path))
                images.append(image)
                print(f"Loaded: {filename} - Shape: {image.shape}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        
        return images
    
    def load_png_labels(self, directory):
        """
        Load PNG label images from directory, filtering out files with underscores.
        
        Loads PNG, JPG, or JPEG files that don't contain underscores in their filenames.
        Files with underscores are assumed to be auxiliary files and are skipped.
        
        Args:
            directory (str): Path to directory containing label files
            
        Returns:
            list: List of numpy arrays representing the loaded label images
            
        Prints:
            - Files being skipped (those with underscores)
            - Number of valid label files found
            - Loading progress for each file
            - Shape information for each loaded label
            - Error messages for files that fail to load
        """
        labels = []
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return labels
        
        png_files = []
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                if "_" not in filename:
                    png_files.append(filename)
                else:
                    print(f"Skipping file with underscore: {filename}")
        
        png_files = sorted(png_files, key=self.natural_key)
        print(f"Found {len(png_files)} valid label files")
        
        for filename in png_files:
            file_path = os.path.join(directory, filename)
            try:
                image = Image.open(file_path)
                labels.append(np.array(image))
                print(f"Loaded: {filename} - Shape: {np.array(image).shape}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return labels
    
    def load_data(self):
        """
        Load both input images and segmentation labels from their respective directories.
        
        This method orchestrates the loading of both TIFF input images and PNG label images,
        then performs a consistency check to ensure equal numbers of images and labels.
        
        Updates:
            self.input_images: List of loaded input images
            self.segmentation_labels: List of loaded segmentation labels
            
        Prints:
            - Loading progress for both image types
            - Data consistency check results
            - Sample shape information
            - Warning if image and label counts don't match
        """
        print("Loading input images...")
        self.input_images = self.load_tiff_images(self.images_dir)
        print(f"\nLoaded {len(self.input_images)} input images")
        
        print("\nLoading segmentation labels...")
        self.segmentation_labels = self.load_png_labels(self.labels_dir)
        print(f"\nLoaded {len(self.segmentation_labels)} label images")
        
        # Verify data consistency
        if len(self.input_images) != len(self.segmentation_labels):
            print(f"WARNING: Mismatch between images ({len(self.input_images)}) and labels ({len(self.segmentation_labels)})")
        else:
            print(f"✓ Data consistency check passed: {len(self.input_images)} image-label pairs")
        
        if self.input_images and self.segmentation_labels:
            print(f"\nSample input image shape: {self.input_images[0].shape}")
            print(f"Sample label image shape: {self.segmentation_labels[0].shape}")
    
    def normalize_images(self):
        """
        Normalize input images using MinMaxScaler to range [0, 1].
        
        Each image is independently normalized using sklearn's MinMaxScaler.
        The normalization is applied per-pixel across all channels, preserving
        the spatial structure while ensuring consistent value ranges.
        
        Updates:
            self.scaled_input_images: List of normalized input images
            
        Prints:
            - Normalization progress
            - Min/max values for first 3 images (for verification)
            - Total number of images normalized
            
        Note:
            This method assumes multi-channel images where the last dimension
            represents channels. For single-channel images, this still works correctly.
        """
        normalized_images = []
        
        print("Normalizing images...")
        for i, image_array in enumerate(self.input_images):
            original_shape = image_array.shape
            reshaped_array = image_array.reshape(-1, original_shape[-1])
            
            scaler = MinMaxScaler()
            scaled_array = scaler.fit_transform(reshaped_array)
            normalized_image = scaled_array.reshape(original_shape)
            normalized_images.append(normalized_image)
            
            if i < 3:
                print(f"Image {i}: min={normalized_image.min():.3f}, max={normalized_image.max():.3f}")
        
        self.scaled_input_images = normalized_images
        print(f"✓ Normalized {len(self.scaled_input_images)} images")
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split dataset into training, validation, and test sets.
        
        Performs a deterministic split based on the provided ratios. The split
        maintains the original order of the data, so ensure data is shuffled
        beforehand if random splitting is desired.
        
        Args:
            train_ratio (float): Proportion of data for training (default: 0.8)
            val_ratio (float): Proportion of data for validation (default: 0.1)
            test_ratio (float): Proportion of data for testing (default: 0.1)
            
        Returns:
            tuple: A tuple containing:
                - (train_images, val_images, test_images): Split input images
                - (train_labels, val_labels, test_labels): Split segmentation labels
                
        Raises:
            AssertionError: If the ratios don't sum to 1.0 (within floating point tolerance)
            
        Prints:
            - Size of each split set
            
        Example:
            >>> images, labels = handler.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
            >>> train_imgs, val_imgs, test_imgs = images
            >>> train_labels, val_labels, test_labels = labels
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        def split_data(dataset):
            total_samples = len(dataset)
            train_len = int(total_samples * train_ratio)
            val_len = int(total_samples * val_ratio)
            
            train_data = dataset[:train_len]
            val_data = dataset[train_len:train_len + val_len]
            test_data = dataset[train_len + val_len:]
            
            return train_data, val_data, test_data
        
        print("Splitting dataset...")
        train_images, val_images, test_images = split_data(self.scaled_input_images)
        train_labels, val_labels, test_labels = split_data(self.segmentation_labels)
        
        print(f"Training set: {len(train_images)} samples")
        print(f"Validation set: {len(val_images)} samples")
        print(f"Test set: {len(test_images)} samples")
        
        return (train_images, val_images, test_images), (train_labels, val_labels, test_labels)
    
    def process_all(self):
        """
        Execute the complete data processing pipeline.
        
        This convenience method runs the entire pipeline in sequence:
        1. Load input images and labels
        2. Normalize input images
        3. Split dataset into train/validation/test sets
        
        Returns:
            tuple: Same as split_dataset() - split images and labels
            
        Example:
            >>> handler = WaterSegmentationDataHandler("/path/to/data")
            >>> (train_imgs, val_imgs, test_imgs), (train_labels, val_labels, test_labels) = handler.process_all()
            
        Note:
            This method uses default split ratios (0.8/0.1/0.1). Use individual methods
            for custom ratios or more control over the pipeline.
        """
        self.load_data()
        self.normalize_images()
        return self.split_dataset()


# Usage Example and Testing
if __name__ == "__main__":
    """
    Example usage of the WaterSegmentationDataHandler class.
    
    This section demonstrates how to use the class for a typical workflow.
    Uncomment and modify the data_path to test with your dataset.
    """
    
    # Example usage
    # data_path = "/path/to/your/water_segmentation_dataset"
    # handler = WaterSegmentationDataHandler(data_path)
    
    # Option 1: Process everything at once
    # (train_imgs, val_imgs, test_imgs), (train_labels, val_labels, test_labels) = handler.process_all()
    
    # Option 2: Step-by-step processing for more control
    # handler.load_data()
    # handler.normalize_images()
    # images, labels = handler.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    # Print results
    # print(f"\nFinal dataset summary:")
    # print(f"Training: {len(images[0])} image-label pairs")
    # print(f"Validation: {len(images[1])} image-label pairs")
    # print(f"Test: {len(images[2])} image-label pairs")
    
    pass