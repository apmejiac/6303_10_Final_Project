import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tensorflow as tf
import torch
from collections import Counter
import random
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#Data loading
class CancerDataset(Dataset):
    """
        PyTorch dataset class for loading and preprocessing medical images related to cancer classification.

        Parameters:
            - dataset_path (str): The path to the root directory of the dataset containing subdirectories for different classes of images (benign, malignant, normal). Source Kaggle

        Methods:
            - __init__(self, dataset_path): Initializes the dataset with the provided path and loads image files and labels using the load_data method.
            - load_data(self): Helper method to load file paths and labels for each image in the dataset.
            - __len__(self): Returns the total number of images in the dataset.
            - __getitem__(self, idx): Loads and preprocesses the image and its corresponding label at the specified index.
        Return:
            - img: Images
            - label: Labels
    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.image_files, self.labels = self.load_data()
        self.transform = transform
    def load_data(self):
        image_files = []
        labels = []
        for class_name in ['benign', 'malignant', 'normal']:
            class_path = os.path.join(self.dataset_path, class_name)
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                image_files.append(image_path)
                labels.append(class_name)
        return image_files, labels
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        # Load the image
        img = Image.open(img_path).convert('RGB')
        # Tranformation to tensor
        if self.transform:
            transform = self.transform
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        img = transform(img)
        # Convert labels to numerical values
        label_encoder = {'benign': 0, 'malignant': 1, 'normal': 2}
        label = label_encoder[label]
        return img, label

    def split_dataset(self, test_size=0.3, validation_size=0.2, random_seed=42):
        """
        Split the dataset into training, validation, and test sets.
        Parameters:
            - test_size (float): Proportion of the dataset to include in the test split.
            - validation_size (float): Proportion of the dataset to include in the validation split.
            - random_seed (int): Seed for reproducibility.
        Returns:
            - train_dataset (CancerDatasetSubset): Training dataset.
            - validation_dataset (CancerDatasetSubset): Validation dataset.
            - test_dataset (CancerDatasetSubset): Test dataset.
        """
        random.seed(random_seed)
        # Split indices
        total_samples = len(self)
        test_size = int(test_size * total_samples)
        validation_size = int(validation_size * total_samples)
        test_indices = random.sample(range(total_samples), test_size)
        remaining_indices = list(set(range(total_samples)) - set(test_indices))
        validation_indices = random.sample(remaining_indices, validation_size)
        train_indices = list(set(remaining_indices) - set(validation_indices))
        # Create subsets
        train_dataset = CancerDatasetSubset(self, train_indices)
        validation_dataset = CancerDatasetSubset(self, validation_indices)
        test_dataset = CancerDatasetSubset(self, test_indices)
        return train_dataset, validation_dataset, test_dataset

class CancerDatasetSubset(Dataset):
    def __init__(self, parent_dataset, indices):
        self.parent_dataset = parent_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.parent_dataset[original_idx]

#EDA
##Added pink graph to follow standards of cancer color
class EDA:
    """
        Exploratory Data Analysis (EDA) class for visualizing and understanding the characteristics of a medical image dataset.
    Methods:
        - plot_class_distribution(self): Visualizes the distribution of classes in the dataset using a bar chart.
        - plot_sample_images(self): Plots sample images from each class in a grid for visual inspection.
        - plot_data_augmentation_impact(self): Visualizes the impact of data augmentation by comparing original and augmented images.
    """
    def __init__(self, dataset_path):
        self.dataset = CancerDataset(dataset_path)

    def plot_class_distribution(self):
        # Count the occurrences of each class
        label_counter = Counter(self.dataset.labels)

        # Extract class names and counts
        classes, counts = zip(*label_counter.items())

        # Plot the bar chart
        plt.bar(classes, counts, color=['pink', 'crimson', 'slategray'])
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution in the Dataset')
        plt.show()

    def plot_sample_images(self):
        benign_samples = [img for img, label in self.dataset if label == 0][:5]
        malignant_samples = [img for img, label in self.dataset if label == 1][:5]
        normal_samples = [img for img, label in self.dataset if label == 2][:5]

        # Visualize the sample images
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        for i, samples in enumerate([benign_samples, malignant_samples, normal_samples]):
            for j, img in enumerate(samples):
                axes[i, j].imshow(transforms.ToPILImage()(img))
                axes[i, j].axis('off')
                axes[i, j].set_title(f'Class {i}')
        plt.show()

    def plot_data_augmentation_impact(self):  #Consider including after data augmentation
        original_image = self.dataset[0][0]
        augmented_images = [transforms.ToPILImage()(self.dataset[0][0]) for _ in range(5)]

        # Visualizes the original and augmented images
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        axes[0].imshow(transforms.ToPILImage()(original_image))
        axes[0].set_title('Original')
        axes[0].axis('off')

        for i, img in enumerate(augmented_images):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f'Augmented {i + 1}')
            axes[i + 1].axis('off')

        plt.show()

##Data augmentation considering unbalance data

def get_balanced_augmentation_transform(horizontal_flip=False, vertical_flip=False, rotation_angle=0,
                                        brightness_range=None, contrast_range=None, saturation_range=None,
                                        normalize=False):
    """
    Generate a composed transformation with balanced augmentation based on the specified augmentations and normalization.

    Parameters:
        - horizontal_flip (bool): Apply random horizontal flip.
        - vertical_flip (bool): Apply random vertical flip.
        - rotation_angle (float): Apply random rotation with the given angle (in degrees).
        - brightness_range (tuple): Range for random changes in brightness, e.g., (0.8, 1.2).
        - contrast_range (tuple): Range for random changes in contrast, e.g., (0.8, 1.2).
        - saturation_range (tuple): Range for random changes in saturation, e.g., (0.8, 1.2).
        - normalize (bool): Apply normalization to the image.

    Returns:
        - transform (Compose): Composed transformation to be applied to the images.
    """
    augmentations = []

    # Add selected augmentations
    if horizontal_flip:
        augmentations.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        augmentations.append(transforms.RandomVerticalFlip())
    if rotation_angle != 0:
        augmentations.append(transforms.RandomRotation(degrees=rotation_angle))
    if brightness_range is not None:
        augmentations.append(transforms.ColorJitter(brightness=brightness_range))
    if contrast_range is not None:
        augmentations.append(transforms.ColorJitter(contrast=contrast_range))
    if saturation_range is not None:
        augmentations.append(transforms.ColorJitter(saturation=saturation_range))

    # Adding standard transforms 
    augmentations.append(transforms.Resize((224, 224)))
    augmentations.append(transforms.ToTensor())
    
    # Add normalization if requested
    if normalize:
        augmentations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = transforms.Compose(augmentations)

    return transform

def balanced_augmentation(img, label, transform, class_counts):
    """
    Apply balanced augmentation to an image based on the specified transformation and class distribution.

    Parameters:
        - img (PIL Image): The input image.
        - label (int): The class label of the image.
        - transform (Compose): Composed transformation to be applied to the images.
        - class_counts (Counter): Class distribution counter.

    Returns:
        - img (Tensor): The augmented image.
        - label (int): The original class label.
    """

    max_count = max(class_counts.values())
    augmentation_prob = {class_label: max_count / count if count > 0 else 0 for class_label, count in class_counts.items()}
    apply_augmentation = random.uniform(0, 1) < augmentation_prob.get(label, 0)

    # Apply data augmentation only if determined to do so
    if apply_augmentation:
        img = transform(img)

    return img, label
