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
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter

#Data loading

class CancerDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_files, self.labels = self.load_data()

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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        img = transform(img)

        # Convert labels to numerical values
        label_encoder = {'benign': 0, 'malignant': 1, 'normal': 2}
        label = label_encoder[label]

        return img, label

#EDA
##Added pink graph to follow standards of cancer color
class EDA:
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

#Preprocessing

#adding train,test,validation split and augmentation fn
