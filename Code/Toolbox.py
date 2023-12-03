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
from torchvision.transforms.functional import resize
import random
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


class DatasetCreator:
    """
    Creates two datasets to be able to be able to overlayed mask and unmask images
    Return:
    - create_dataset_mask_folder: Dataset with mask images
    - create_images_without_mask_folder: Dataset with original images no mask
    """

    def __init__(self, data_path):
        self.data_path = data_path

    #Function to generate mask dataset
    def create_dataset_mask_folder(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        for class_name in ['benign', 'malignant', 'normal']:
            class_path = os.path.join(self.data_path, class_name)
            output_class_path = os.path.join(output_path, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            for filename in os.listdir(class_path):
                if filename.endswith('.png'):
                    mask_filename = filename.replace('.png', '_mask.png')
                    mask_path = os.path.join(class_path, mask_filename)
                    if not os.path.exists(mask_path):
                        image_path = os.path.join(class_path, filename)
                        output_image_path = os.path.join(output_class_path, filename)
                        shutil.copy(image_path, output_image_path)
    #Function to separate files without mask
    def create_images_without_mask_folder(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        for class_name in ['benign', 'malignant', 'normal']:
            class_path = os.path.join(self.data_path, class_name)
            output_class_path = os.path.join(output_path, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            for filename in os.listdir(class_path):
                if filename.endswith('.png') and 'mask' not in filename.lower():
                    image_path = os.path.join(class_path, filename)
                    output_image_path = os.path.join(output_class_path, filename)
                    shutil.copy(image_path, output_image_path)

##Function to create overlayed images with mask to test models with this dataset
def overlay_images_with_masks(input_path_mask, input_path_without_mask, output_path):
    """
    Creates  overlayed images
    Return:
    - overlay_images_with_masks: Dataset with overlayed images
    """
    os.makedirs(output_path, exist_ok=True)
    classes = ['benign', 'malignant', 'normal']
    for class_name in classes:
        class_path_mask = os.path.join(input_path_mask, class_name)
        class_path_without_mask = os.path.join(input_path_without_mask, class_name)
        output_class_path = os.path.join(output_path, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        for filename in os.listdir(class_path_without_mask):
            if filename.endswith('.png'):
                image_path_without_mask = os.path.join(class_path_without_mask, filename)
                mask_filename = filename.replace('.png', '_mask.png')
                mask_path = os.path.join(class_path_mask, mask_filename)
                if os.path.exists(mask_path):
                    image_without_mask = Image.open(image_path_without_mask)
                    mask = Image.open(mask_path)
                    if image_without_mask.mode != mask.mode:
                        mask = mask.convert(image_without_mask.mode)
                    if image_without_mask.size != mask.size:
                        mask = mask.resize(image_without_mask.size)
                    overlayed = Image.blend(image_without_mask, mask, alpha=0.5)
                    output_image_path = os.path.join(output_class_path, filename.replace('.png', '_overlay.png'))
                    overlayed.save(output_image_path)
                    # print(f"Overlayed Image Saved: {output_image_path}")


class CancerDataset(Dataset):
    """
    PyTorch dataset class for loading and preprocessing medical images related to cancer classification.

    Parameters:
        - dataset_path (str): The path to the root directory of the dataset containing subdirectories for different classes of images (benign, malignant, normal). Source Kaggle

    Methods:
        - __init__(self, dataset_path, dataset_type='original', transform=None): Initializes the dataset with the provided path and loads image files and labels using the load_data method.
        - load_data(self, dataset_type): Helper method to load file paths and labels for each image in the dataset.
        - __len__(self): Returns the total number of images in the dataset.
        - __getitem__(self, idx): Loads and preprocesses the image and its corresponding label at the specified index.
    Return:
        - img: Images
        - label: Labels
    """

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        # self.dataset_type = dataset_type
        self.image_files, self.labels =self.load_data()
        self.transform = transform

    def load_data(self):
        image_files = []
        labels = []

        for class_name in ['benign', 'malignant', 'normal']:
            class_path = os.path.join(self.data_path, class_name)
            # print(f"Loading class: {class_name}, Path: {class_path}")
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                image_files.append(image_path)
                labels.append(class_name)
                # print(f"Loaded: {image_path}, Label: {class_name}")

        return image_files, labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # Tranformation to tensor ##### Triple check if we want to use this size or another

        # transform = transforms.Compose([
        #             transforms.Resize((224, 224)),
        #             transforms.ToTensor(),
        #             transforms.Lambda(lambda x: x / 255.0)  # setting scale from 0 to 1
        #         ])
        #
        if self.transform:
            img = self.transform(img)

        label_encoder = {'benign': 0, 'malignant': 1, 'normal': 2}
        label = label_encoder[label]

        return img, label

    def split_dataset(self, test_size=0.3, validation_size=0.2, random_seed=42):
        total_samples = len(self)
        test_size = int(test_size * total_samples)
        validation_size = int(validation_size * total_samples)
        test_indices = random.sample(range(total_samples), test_size)
        remaining_indices = list(set(range(total_samples)) - set(test_indices))
        validation_indices = random.sample(remaining_indices, validation_size)
        train_indices = list(set(remaining_indices) - set(validation_indices))

        train_dataset = CancerDataset(self.data_path, transform=self.transform)
        train_dataset.image_files = [train_dataset.image_files[i] for i in train_indices]
        train_dataset.labels = [train_dataset.labels[i] for i in train_indices]

        validation_dataset = CancerDataset(self.data_path, transform=self.transform)
        validation_dataset.image_files = [validation_dataset.image_files[i] for i in validation_indices]
        validation_dataset.labels = [validation_dataset.labels[i] for i in validation_indices]

        test_dataset = CancerDataset(self.data_path, transform=self.transform)
        test_dataset.image_files = [test_dataset.image_files[i] for i in test_indices]
        test_dataset.labels = [test_dataset.labels[i] for i in test_indices]

        return train_dataset, validation_dataset, test_dataset
# class CancerDatasetSubset(Dataset):
#     def __init__(self, parent_dataset, indices):
#         self.parent_dataset = parent_dataset
#         self.indices = indices
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, idx):
#         original_idx = self.indices[idx]
#         return self.parent_dataset[original_idx]
# # EDA
# ##Added pink graph to follow standards of cancer color
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

    def plot_class_distribution(self, custom_title=None):
        # Count the occurrences of each class
        label_counter = Counter(self.dataset.labels)

        # Extract class names and counts
        classes, counts = zip(*label_counter.items())

        # Plot the bar chart
        plt.bar(classes, counts, color=['pink', 'crimson', 'slategray'])
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')

        # Set the title based on the provided argument or use a default title
        title = custom_title if custom_title else 'Class Distribution in the Dataset'
        plt.title(title)

        for i, count in enumerate(counts):
            plt.text(classes[i], count / 2, str(count), ha='center', va='center', color='black', fontweight='bold')

        plt.show()
    #
    def plot_sample_images(self, image_size=(224, 224)):
        benign_samples = [img for img, label in self.dataset if label == 0][:5]
        malignant_samples = [img for img, label in self.dataset if label == 1][:5]
        normal_samples = [img for img, label in self.dataset if label == 2][:5]

        # Resize images to a specific size
        transform = transforms.Compose([transforms.Resize(image_size)])

        # Visualize the sample images
        fig, axes = plt.subplots(3, 5, figsize=(15, 9))
        for i, samples in enumerate([benign_samples, malignant_samples, normal_samples]):
            for j, img in enumerate(samples):
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)
                img = transform(img)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                axes[i, j].set_title(f'Class {i}')
        plt.show()
#     def plot_data_augmentation_impact(self):  # Consider including after data augmentation
#         original_image = self.dataset[0][0]
#         augmented_images = [transforms.ToPILImage()(self.dataset[0][0]) for _ in range(5)]
#
#         # Visualizes the original and augmented images
#         fig, axes = plt.subplots(1, 6, figsize=(15, 5))
#         axes[0].imshow(transforms.ToPILImage()(original_image))
#         axes[0].set_title('Original')
#         axes[0].axis('off')
#
#         for i, img in enumerate(augmented_images):
#             axes[i + 1].imshow(img)
#             axes[i + 1].set_title(f'Augmented {i + 1}')
#             axes[i + 1].axis('off')
#
#         plt.show()
#
#     # Add something to find average and std for normalization of images
#
#
##Data augmentation considering unbalance data
#
# Gaussian Noise
class GaussianNoise(object):
    def __init__(self, std=0.001):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        noisy_tensor = tensor + noise

        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

class CustomAugmentedDataset(Dataset):
    def __init__(self, original_dataset, augment_transform, classes_to_augment=None):
        self.original_dataset = original_dataset
        self.augment_transform = augment_transform
        self.classes_to_augment = classes_to_augment

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        original_item = self.original_dataset[index]
        image, label = original_item#['image']#, original_item['label']

        # Check if augmentation is needed for this class
        if self.classes_to_augment is None or label in self.classes_to_augment:
            augmented_image = self.augment_transform(image)
            return augmented_image, label  # Return as tuple
        else:
            return image, label
        #     return {'image': augmented_image, 'label': label}
        # else:
        #     return {'image': transforms.ToTensor()(image), 'label': label}

def apply_augmentation_to_dataset(original_dataset, horizontal_flip=False, vertical_flip=False,
                                  rotation_angle=0, brightness_range=None, contrast_range=None,
                                  saturation_range=None, normalize=False, gaussian_noise=False,  classes_to_augment=None):
    """
    Apply data augmentation to a training dataset.
    Parameters:
        - original_dataset (Dataset): The original training dataset.
        - horizontal_flip (bool): Apply random horizontal flip.
        - vertical_flip (bool): Apply random vertical flip.
        - rotation_angle (float): Apply random rotation with the given angle (in degrees).
        - brightness_range (tuple): Range for random changes in brightness, e.g., (0.8, 1.2).
        - contrast_range (tuple): Range for random changes in contrast, e.g., (0.8, 1.2).
        - saturation_range (tuple): Range for random changes in saturation, e.g., (0.8, 1.2).
        - normalize (bool): Apply normalization to the image.
        - classes_to_augment (list or None): List of classes to apply augmentation. If None, augment all classes.
    Returns:
        - augmented_dataset (Dataset): Augmented training dataset.
    """
    augmentations = []
    # Add selected augmentations
    if horizontal_flip:
        augmentations.append(transforms.RandomHorizontalFlip())
    if vertical_flip:
        augmentations.append(transforms.RandomVerticalFlip())
    if rotation_angle != 0:
        augmentations.append(transforms.RandomRotation(degrees=rotation_angle))
    if gaussian_noise is not None:
        augmentations.append(GaussianNoise())
    if brightness_range is not None:
        augmentations.append(transforms.ColorJitter(brightness=brightness_range))
    if contrast_range is not None:
        augmentations.append(transforms.ColorJitter(contrast=contrast_range))
    if saturation_range is not None:
        augmentations.append(transforms.ColorJitter(saturation=saturation_range))
    # Add normalization if requested
    if normalize:
        augmentations.append(transforms.Normalize((0.3229, 0.3229, 0.3228), (0.1995, 0.1995, 0.1994)))
    augment_transform = transforms.Compose(augmentations)
    # Create augmented dataset
    augmented_dataset = CustomAugmentedDataset(original_dataset, augment_transform, classes_to_augment)
    return augmented_dataset

def custom_collate_fn(batch):
    images, labels = zip(*batch)

    # Assuming images are PIL Image objects, convert them to torch tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Choose the desired size
        transforms.ToTensor(),
        # Add any other transformations you need
    ])

    images = [transform(img) for img in images]
    labels = torch.tensor(labels)  # Convert labels to a tensor

    # Stack images into a single tensor
    images = torch.stack(images)

    return images, labels
