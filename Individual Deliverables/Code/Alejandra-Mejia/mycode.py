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

    # Function to generate mask dataset
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

    # Function to separate files without mask
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
        self.image_files, self.labels = self.load_data()
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

    def get_ids_and_labels(self):
        return self.image_files, self.labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # Tranformation to tensor ##### Triple check if we want to use this size or another

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            # transforms.Lambda(lambda x: x / 255.0)  # setting scale from 0 to 1
        ])
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

        plt.savefig("class_distribution_plot.png")

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

        plt.savefig("sample_images.png")

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
        image, label = original_item  # ['image']#, original_item['label']

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
                                  saturation_range=None, normalize=False, gaussian_noise=False,
                                  classes_to_augment=None):
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

from Toolbox import DatasetCreator ,overlay_images_with_masks, CancerDataset, EDA,apply_augmentation_to_dataset, custom_collate_fn, predict
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
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
import pandas as pd

# Getting Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

np.random.seed(42)

# Set random seed for Python's built-in random module
random.seed(42)

# Set random seed for PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed(42)  # If using CUDA
torch.backends.cudnn.deterministic = True

# Loading separated and overlayed datasets

path = "/home/ubuntu/DL_Proj/Data/"
# # path = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data"
output_path_mask = "/home/ubuntu/DL_Proj/ImagesMask/"
output_path_without_mask = "/home/ubuntu/DL_Proj/ImagesWithoutMask/"
output_path_overlay = "/home/ubuntu/DL_Proj/ImagesOverlay/"

dataset_creator = DatasetCreator(path)

#Creates dataset on virtual machine (comment it out after having them on virtual machine)
dataset_creator.create_dataset_mask_folder(output_path_mask)
dataset_creator.create_images_without_mask_folder(output_path_without_mask)
overlay_images_with_masks(output_path_mask, output_path_without_mask, output_path_overlay)

#========================================================================================================

### Complete dataset EDA
eda_C = EDA(path)
eda_C.plot_class_distribution(custom_title='Class distribution raw dataset')
eda_C.plot_sample_images()


#========================================================================================================
# #Dataset with only original images
cancer_dataset_original = CancerDataset(output_path_without_mask)
train_dataset, validation_dataset, test_dataset = cancer_dataset_original.split_dataset(test_size=0.3, validation_size=0.2)
# #
# # # Create DataLoader instances for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False,collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)

#Mean and sd for normalization
mean = 0.0
std = 0.0
total_samples = 0

for data, _ in train_loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    total_samples += batch_samples

mean /= total_samples
std /= total_samples

print(f"Calculated mean: {mean}, std: {std}")

# #EDA prior augmentation-original images - Commenting out for now

eda = EDA(output_path_without_mask)
eda.plot_class_distribution(custom_title='Class distribution only images no mask')
eda.plot_sample_images()


# #Considering data augmentation for training dataset
minority_classes = ['malignant', 'normal']

# Apply augmentation to the minority classes in the dataset
augmented_dataset = apply_augmentation_to_dataset(train_dataset, horizontal_flip=True,
                                                  vertical_flip=True, rotation_angle=90,
                                                  brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                                                  saturation_range=(0.8, 1.2), normalize=True, classes_to_augment=minority_classes)
num_epochs = 20
#
def model_definition():
    # Defining Model
    model = models.resnet50(weights=True)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(device)

    # Defining Criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # # Consider using this scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    return model, optimizer, criterion
#
def train_model(train_loader, validation_loader, num_epochs, save_on=True):
    # Getting model
    model, optimizer, criterion = model_definition()
    f1_best = 0

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Beggining Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps_train = 0

        with tqdm(total=len(train_loader), desc="Epoch {}".format(epoch + 1)) as pbar:
            for images, labels in train_loader:
                # Getting to GPU
                images, labels = images.to(device), labels.to(device)

                # Apply data augmentation to the batch
                augmented_batch = [augmented_dataset[i] for i in range(len(images))]
                augmented_images, augmented_labels = zip(*augmented_batch)
                augmented_images, augmented_labels = custom_collate_fn(list(zip(augmented_images, augmented_labels)))

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps_train += 1

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(total_loss / steps_train))

            average_loss = total_loss / len(train_loader)
            train_losses.append(average_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        # Validation phase for the original dataset
        model.eval()
        val_loss = 0
        steps_val = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                steps_val += 1

            average_val_loss = val_loss / len(validation_loader)
            val_losses.append(average_val_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {average_val_loss:.4f}')

            # Save the model if validation loss improves
            if save_on and average_val_loss < f1_best:
                torch.save(model, "trained_model.pt")
                f1_best = average_val_loss
                print("The model has been saved and updated!")

    # Plotting the training and validation loss curves
    # Plotting the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, num_epochs), train_losses, label='Training Loss', color='orchid')
    plt.plot(range(0, num_epochs), val_losses, label='Validation Loss', color='hotpink')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.ylim(0, 5)  # Set the y-axis limit to 5
    plt.yticks([i * 0.5 for i in range(0, 11)])  # Adjust y-axis ticks accordingly
    plt.xticks(range(0, num_epochs))  # Set x-axis ticks based on the actual number of epochs
    plt.legend()
    plt.show()

    return model

# ## Applying model to augment data
model = train_model(train_loader, validation_loader, num_epochs)
# #
## Printing Metrics for the test data
test_pred = []
test_labels = []

# Getting predictions
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        test_pred.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

#========================================================================================================

#Dataset overlayed images (original + masks)
cancer_dataset_overlayed= CancerDataset(output_path_overlay)
train_dataset_o, validation_dataset_o, test_dataset_o = cancer_dataset_overlayed.split_dataset(test_size=0.3, validation_size=0.2)
# image_ids, image_labels = test_dataset_o.get_ids_and_labels()

# # Create DataLoader instances for training, validation, and test sets
train_loader_o = DataLoader(train_dataset_o, batch_size=32, shuffle=True,collate_fn=custom_collate_fn)
validation_loader_o = DataLoader(validation_dataset_o, batch_size=32, shuffle=False,collate_fn=custom_collate_fn)
test_loader_o = DataLoader(test_dataset_o, batch_size=32, shuffle=False,collate_fn=custom_collate_fn)

# #EDA prior augmentation-overlayed images - Commenting out for now

eda_o = EDA(output_path_overlay)
eda_o.plot_class_distribution(custom_title='Class distribution overlayed images')
eda_o.plot_sample_images()
# #
#

augmented_dataset_o = apply_augmentation_to_dataset(train_dataset_o, horizontal_flip=True,
                                                  vertical_flip=True, rotation_angle=90,
                                                  brightness_range=(0.2, 0.8), contrast_range=(0.2, 0.8),
                                                  saturation_range=(0.2, 0.8), normalize=True, classes_to_augment=minority_classes)
# num_epochs = 20

def model_definition():
    # Defining Model
    model = models.resnet50(weights=True)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(device)

    # Defining Criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # # Consider using this scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)

    return model, optimizer, criterion

def train_model_and_plot(train_loader, validation_loader, num_epochs, save_on=True):
    # Getting model
    model, optimizer, criterion = model_definition()
    f1_best = float('-inf')  # Initialize f1_best to negative infinity

    # Lists to store training and validation losses
    train_losses = []
    val_losses = []

    # Beggining Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps_train = 0

        with tqdm(total=len(train_loader), desc="Epoch {}".format(epoch + 1)) as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Apply data augmentation to the batch
                augmented_batch = [augmented_dataset_o[i] for i in range(len(images))]
                augmented_images, augmented_labels = zip(*augmented_batch)
                augmented_images, augmented_labels = custom_collate_fn(list(zip(augmented_images, augmented_labels)))

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps_train += 1

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(total_loss / steps_train))

            average_loss = total_loss / len(train_loader)
            train_losses.append(average_loss)

        # Validation phase for the original dataset
        model.eval()
        val_loss = 0
        steps_val = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                steps_val += 1

                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        average_val_loss = val_loss / len(validation_loader)
        val_losses.append(average_val_loss)

        # Print and save losses
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

        # Save the model if validation loss improves
        if save_on and average_val_loss > f1_best:
            torch.save(model, "trained_model.pt")
            f1_best = average_val_loss
            print("The model has been saved and updated!")

    # Plotting the training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='orchid')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='hotpink')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    # Explicitly set the y-axis limit
    plt.ylim(0, 15)
    plt.gca().set_ylim(0, 15)

    plt.yticks([i * 0.5 for i in range(0, 31)])  # Adjust y-axis ticks accordingly
    plt.xticks(range(0, num_epochs + 1))  # Set x-axis ticks based on the actual number of epochs
    plt.legend()
    plt.show()
    return model
#
# ## Applying model to augment data
model = train_model_and_plot(train_loader_o, validation_loader_o, num_epochs)
# #



cm = confusion_matrix(test_labels_o, test_pred_o)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix Original data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
