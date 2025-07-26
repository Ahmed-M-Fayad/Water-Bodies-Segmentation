# -*- coding: utf-8 -*-
"""
Water Segmentation Pipeline v2 - Class-based Implementation
Complete pipeline for processing multispectral water segmentation data
"""

import os
import re
import zipfile
import numpy as np
from PIL import Image
import tifffile as tiff
import matplotlib.pyplot as plt
import time


class WaterSegmentationPipeline:
    """
    Complete water segmentation pipeline for multispectral data processing
    """

    def __init__(self, zip_path=None, extract_path=None, selected_channels=None):
        """
        Initialize the pipeline

        Args:
            zip_path (str): Path to the .zip dataset file
            extract_path (str): Directory where the zip content will be extracted
            selected_channels (list): List of channel indices to select (default: [2, 3, 4, 5, 6, 10])
        """
        self.zip_path = zip_path
        self.extract_path = extract_path
        self.selected_channels = selected_channels or [2, 3, 4, 5, 6, 10]

        # Data storage
        self.images = []
        self.labels = []
        self.channel_stats = {}
        self.filtered_images = []
        self.normalized_images = []

        # Pipeline state
        self.data_loaded = False
        self.stats_computed = False
        self.channels_filtered = False
        self.data_normalized = False

    def inspect_extracted_data(self, extract_to=None):
        """
        Inspect the extracted dataset structure to understand folder organization

        Args:
            extract_to (str): Directory to inspect
        """
        extract_to = extract_to or self.extract_path

        if not extract_to or not os.path.exists(extract_to):
            print(f"Directory not found: {extract_to}")
            return

        print(f"\n📁 Inspecting directory structure: {extract_to}")
        print("=" * 50)

        for root, dirs, files in os.walk(extract_to):
            level = root.replace(extract_to, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")

            # Show first few files in each directory
            sub_indent = " " * 2 * (level + 1)
            for i, file in enumerate(files[:5]):  # Show first 5 files
                print(f"{sub_indent}{file}")
            if len(files) > 5:
                print(f"{sub_indent}... and {len(files) - 5} more files")
        print("=" * 50)

    def prepare_water_segmentation_data(
        self,
        zip_path=None,
        extract_to=None,
        images_subdir="images",
        labels_subdir="labels",
    ):
        """
        Unzips the dataset and loads TIFF images and PNG labels.

        Args:
            zip_path (str): Path to the .zip dataset file (optional if set in __init__)
            extract_to (str): Directory where the zip content will be extracted (optional if set in __init__)
            images_subdir (str): Name of images subdirectory (default: "images")
            labels_subdir (str): Name of labels subdirectory (default: "labels")

        Returns:
            tuple: (input_images, segmentation_labels)
        """
        zip_path = zip_path or self.zip_path
        extract_to = extract_to or self.extract_path

        if not zip_path or not extract_to:
            raise ValueError("zip_path and extract_to must be provided")

        # -------- Unzip dataset --------
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted: {zip_path} → {extract_to}")

        # -------- Inspect structure --------
        self.inspect_extracted_data(extract_to)

        # -------- Helpers --------
        def natural_key(file_name):
            return [
                int(text) if text.isdigit() else text
                for text in re.split(r"(\d+)", file_name)
            ]

        # -------- Auto-detect or use specified directories --------
        def find_files_recursively(base_dir, extensions, skip_underscore=False):
            """Find files with given extensions recursively"""
            found_files = []
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        if skip_underscore and "_" in file:
                            print(f"Skipping file with underscore: {file}")
                            continue
                        found_files.append(os.path.join(root, file))
            return found_files

        # -------- Load TIFF Images --------
        images_dir = os.path.join(extract_to, images_subdir)
        input_images = []

        if not os.path.exists(images_dir):
            print(f"Standard images directory not found: {images_dir}")
            print("🔍 Searching for TIFF files recursively...")
            tiff_files = find_files_recursively(extract_to, [".tif", ".tiff"])
        else:
            tiff_files = []
            for file in os.listdir(images_dir):
                if file.endswith((".tif", ".tiff")):
                    tiff_files.append(os.path.join(images_dir, file))

        tiff_files = sorted(tiff_files, key=lambda x: natural_key(os.path.basename(x)))
        print(f"Found {len(tiff_files)} TIFF files")

        for file_path in tiff_files:
            filename = os.path.basename(file_path)
            try:
                image = np.array(tiff.imread(file_path))
                input_images.append(image)
                print(f"Loaded image: {filename} - Shape: {image.shape}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        # -------- Load PNG Labels --------
        labels_dir = os.path.join(extract_to, labels_subdir)
        segmentation_labels = []

        if not os.path.exists(labels_dir):
            print(f"Standard labels directory not found: {labels_dir}")
            print("🔍 Searching for label files recursively...")
            label_files = find_files_recursively(
                extract_to, [".png", ".jpg", ".jpeg"], skip_underscore=True
            )
        else:
            label_files = []
            for file in os.listdir(labels_dir):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    if "_" not in file:
                        label_files.append(os.path.join(labels_dir, file))
                    else:
                        print(f"Skipping label with underscore: {file}")

        label_files = sorted(
            label_files, key=lambda x: natural_key(os.path.basename(x))
        )
        print(f"Found {len(label_files)} valid label files")

        for file_path in label_files:
            filename = os.path.basename(file_path)
            try:
                image = Image.open(file_path)
                segmentation_labels.append(np.array(image))
                print(f"Loaded label: {filename} - Shape: {np.array(image).shape}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        # -------- Summary --------
        print(
            f"\n✓ Loaded {len(input_images)} images and {len(segmentation_labels)} labels"
        )
        if len(input_images) != len(segmentation_labels):
            print("⚠️ WARNING: Mismatch in number of images and labels")

        # Store in class attributes
        self.images = input_images
        self.labels = segmentation_labels
        self.data_loaded = True if input_images else False

        return input_images, segmentation_labels

    def analyze_channels(self, image_list=None):
        """
        Compute per-channel statistics (min, max, mean, std) for multispectral images.

        Args:
            image_list (list): List of numpy arrays with shape (H, W, C) (optional if data already loaded)

        Returns:
            dict: Dictionary containing statistics for each channel
        """
        image_list = image_list or self.images

        if not image_list:
            raise ValueError("No images available. Load data first.")

        num_channels = image_list[0].shape[-1]
        stats = {
            i: {"min": [], "max": [], "mean": [], "std": []}
            for i in range(num_channels)
        }

        for image in image_list:
            for c in range(num_channels):
                channel_data = image[:, :, c].flatten()
                stats[c]["min"].append(channel_data.min())
                stats[c]["max"].append(channel_data.max())
                stats[c]["mean"].append(channel_data.mean())
                stats[c]["std"].append(channel_data.std())

        # Aggregate results per channel
        for c in range(num_channels):
            stats[c]["min"] = float(np.min(stats[c]["min"]))
            stats[c]["max"] = float(np.max(stats[c]["max"]))
            stats[c]["mean"] = float(np.mean(stats[c]["mean"]))
            stats[c]["std"] = float(np.mean(stats[c]["std"]))

        # Store in class attributes
        self.channel_stats = stats
        self.stats_computed = True

        return stats

    def display_results_all_channels(
        self, image=None, true_mask=None, pred_mask=None, index=0
    ):
        """Display 12-channel input image as RGB composite, optionally show true/pred masks"""
        image = (
            image
            if image is not None
            else (self.images[index] if self.images else None)
        )

        if image is None:
            raise ValueError("No image available to display")

        # Normalize each channel
        norm_image = (image - image.min(axis=(0, 1), keepdims=True)) / (
            image.max(axis=(0, 1), keepdims=True)
            - image.min(axis=(0, 1), keepdims=True)
            + 1e-6
        )

        # Average 12 channels into 3: R = 0–3, G = 4–7, B = 8–11
        R = np.mean(norm_image[:, :, 0:4], axis=-1)
        G = np.mean(norm_image[:, :, 4:8], axis=-1)
        B = np.mean(norm_image[:, :, 8:12], axis=-1)
        composite = np.stack([R, G, B], axis=-1)

        # Start plotting
        n = 1 + int(true_mask is not None) + int(pred_mask is not None)
        plt.figure(figsize=(5 * n, 5))

        plt.subplot(1, n, 1)
        plt.title(f"Composite 12-Ch Image {index}")
        plt.imshow(composite)
        plt.axis("off")

        if true_mask is not None:
            plt.subplot(1, n, 2)
            plt.title(f"True Mask {index}")
            plt.imshow(true_mask.squeeze(), cmap="gray")
            plt.axis("off")

        if pred_mask is not None:
            plt.subplot(1, n, 3 if true_mask is not None else 2)
            plt.title(f"Predicted Mask {index}")
            plt.imshow(pred_mask.squeeze(), cmap="gray")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_multispectral_channels(
        self, image=None, channel_names=None, cmap="viridis", index=0
    ):
        """
        Plots each channel of a multispectral image separately.

        Args:
            image (np.ndarray): Image array of shape (H, W, C) (optional if data loaded)
            channel_names (list): Optional list of channel names (length C)
            cmap (str): Matplotlib colormap to use for display
            index (int): Index of image to plot if using stored images
        """
        if image is None:
            if self.normalized_images:
                image = self.normalized_images[index]
            elif self.filtered_images:
                image = self.filtered_images[index]
            elif self.images:
                image = self.images[index]
            else:
                raise ValueError("No image available to plot")

        num_channels = image.shape[-1]
        cols = 4
        rows = (num_channels + cols - 1) // cols  # Round up

        plt.figure(figsize=(4 * cols, 4 * rows))
        for i in range(num_channels):
            channel = image[:, :, i]
            plt.subplot(rows, cols, i + 1)
            plt.imshow(channel, cmap=cmap)
            title = f"Channel {i+1}"
            if channel_names and i < len(channel_names):
                title = f"{channel_names[i]}"
            plt.title(title)
            plt.colorbar()
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def filter_channels(self, selected_channels=None):
        """
        Filter images to keep only selected channels

        Args:
            selected_channels (list): List of channel indices to select
        """
        selected_channels = selected_channels or self.selected_channels

        if not self.images:
            raise ValueError("No images loaded. Load data first.")

        self.filtered_images = [img[:, :, selected_channels] for img in self.images]
        self.selected_channels = selected_channels
        self.channels_filtered = True

        print(f"✓ Filtered to channels: {selected_channels}")
        return self.filtered_images

    def normalize_per_channel(
        self, filtered_image_list=None, channel_stats=None, selected_channels=None
    ):
        """
        Normalize each selected channel of each filtered image using precomputed min/max values.

        Args:
            filtered_image_list (list): List of numpy arrays (optional if filtered_images available)
            channel_stats (dict): Channel statistics (optional if already computed)
            selected_channels (list): List of selected channel indices (optional if already set)

        Returns:
            list: Normalized images with shape (H, W, len(selected_channels))
        """
        filtered_image_list = filtered_image_list or self.filtered_images
        channel_stats = channel_stats or self.channel_stats
        selected_channels = selected_channels or self.selected_channels

        if not filtered_image_list:
            raise ValueError("No filtered images available. Filter channels first.")
        if not channel_stats:
            raise ValueError("No channel statistics available. Analyze channels first.")

        normalized_images = []
        num_selected = len(selected_channels)

        for image in filtered_image_list:
            norm_image = np.zeros_like(image, dtype=np.float32)
            for i, original_channel_index in enumerate(selected_channels):
                c_min = channel_stats[original_channel_index]["min"]
                c_max = channel_stats[original_channel_index]["max"]
                if c_max - c_min == 0:
                    norm_image[:, :, i] = 0.0
                else:
                    norm_image[:, :, i] = (image[:, :, i] - c_min) / (c_max - c_min)
            normalized_images.append(norm_image)

        # Store in class attributes
        self.normalized_images = normalized_images
        self.data_normalized = True

        return normalized_images

    def print_channel_stats(self):
        """Print channel statistics in a formatted way"""
        if not self.channel_stats:
            print("No channel statistics available. Run analyze_channels() first.")
            return

        print("\nChannel Statistics:")
        for c, values in self.channel_stats.items():
            print(
                f"Channel {c}: min={values['min']:.2f}, max={values['max']:.2f}, mean={values['mean']:.2f}, std={values['std']:.2f}"
            )

    def get_pipeline_state(self):
        """Get current pipeline state"""
        return {
            "data_loaded": self.data_loaded,
            "stats_computed": self.stats_computed,
            "channels_filtered": self.channels_filtered,
            "data_normalized": self.data_normalized,
            "num_images": len(self.images),
            "num_labels": len(self.labels),
            "selected_channels": self.selected_channels,
            "final_shape": (
                self.normalized_images[0].shape if self.normalized_images else None
            ),
        }

    def run_complete_pipeline(
        self, zip_path=None, extract_path=None, selected_channels=None, show_plots=True
    ):
        """
        Run the complete water segmentation pipeline

        Args:
            zip_path (str): Path to dataset zip file
            extract_path (str): Extraction directory
            selected_channels (list): Channels to select
            show_plots (bool): Whether to show visualization plots

        Returns:
            tuple: (normalized_images, labels, channel_stats, filtered_images)
        """
        print("=" * 60)
        print("WATER SEGMENTATION PIPELINE V2 - CLASS BASED")
        print("=" * 60)

        # Update parameters if provided
        if zip_path:
            self.zip_path = zip_path
        if extract_path:
            self.extract_path = extract_path
        if selected_channels:
            self.selected_channels = selected_channels

        # Step 1: Load data
        print("\n[STEP 1] Loading dataset...")
        self.prepare_water_segmentation_data()

        # Step 2: Analyze channel statistics
        print("\n[STEP 2] Analyzing channel statistics...")
        self.analyze_channels()
        self.print_channel_stats()

        # Step 3: Display original composite
        if show_plots:
            print("\n[STEP 3] Displaying original composite image...")
            if len(self.images) > 0:
                self.display_results_all_channels()

        # Step 4: Plot individual channels
        if show_plots:
            print("\n[STEP 4] Plotting individual channels...")
            if len(self.images) > 1:
                self.plot_multispectral_channels(index=1)

        # Step 5: Filter selected channels
        print(f"\n[STEP 5] Filtering selected channels: {self.selected_channels}")
        self.filter_channels()

        # Step 6: Display filtered channels
        if show_plots:
            print("\n[STEP 6] Displaying filtered channels...")
            self.plot_multispectral_channels()

        # Step 7: Normalize per channel
        print("\n[STEP 7] Normalizing per channel...")
        self.normalize_per_channel()

        # Step 8: Display normalized channels
        if show_plots:
            print("\n[STEP 8] Displaying normalized channels...")
            self.plot_multispectral_channels()

        # Final summary
        state = self.get_pipeline_state()
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"✓ Processed {state['num_images']} images")
        print(f"✓ Processed {state['num_labels']} labels")
        print(
            f"✓ Selected {len(state['selected_channels'])} channels: {state['selected_channels']}"
        )
        print(f"✓ Final image shape: {state['final_shape']}")
        print("=" * 60)

        return (
            self.normalized_images,
            self.labels,
            self.channel_stats,
        )


# Example usage:
# if __name__ == "__main__":
#     # Initialize pipeline
#     pipeline = WaterSegmentationPipeline()
#
#     # Run complete pipeline
#     normalized_images, labels, stats, filtered_images = pipeline.run_complete_pipeline(
#         zip_path="/content/drive/MyDrive/Datasets/data.zip",
#         extract_path="/content/drive/MyDrive/Datasets/data"
#     )
#
#     # Or run steps individually
#     # pipeline = WaterSegmentationPipeline(zip_path="...", extract_path="...")
#     # pipeline.prepare_water_segmentation_data()
#     # pipeline.analyze_channels()
#     # pipeline.filter_channels([2, 3, 4, 5, 6, 10])
#     # pipeline.normalize_per_channel()
#
#     # Access results
#     # print(f"Pipeline state: {pipeline.get_pipeline_state()}")
