### My Code - Jack McMorrow

# Code from streamlit.py
# Imports
import streamlit as st
import torch
import io
from PIL import Image
from Toolbox import predict
import pandas as pd
# from Toolbox import predict_model # -- create model prediction file

# Creating different tabs for the presentation
def overview_tab():
    """
    Showing overview of the project and applications
    """
    
    st.title("Breast Cancer Image Classification Model")
    
    
    st.image("Breast_fibroadenoma.gif")
    
    gif_path = "CANCER.GIF"  # Replace with the actual path to your GIF file

    # Inject custom CSS and HTML to position the GIF in the upper right corner
    # st.image(gif_path,  style='position: fixed; top: 10px; right: 10px;')
    st.markdown(
        """
        <style>
            .upper-right {
                position: fixed;
                top: 10px;
                right: 10px;
            }
        </style>
        <div class="upper-right">
            <img src="CANCER.GIF" alt="Animated GIF" width="100"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ---------------------------------------------------
    # Text
    text = "Ultrasound images can be used as a screening technique to diagnose breast cancer. They are highly accurate, with around 80% of breast cancers being detected"
    st.write(text)
    
    text = "Data is collected from Kaggle and contains 780 images labelled as Normal, Benign, or Malignant"
    st.write(text)
    
    text = "Our goal was to use machine learning to correctly classify these images into their respective categories"
    st.write(text)
    
    st.divider()
    # Upload example images, including regular, mask, and overlayed images
    st.write("Distribution of classes in original dataset")
    class_distribtion = Image.open("class_distribution_plot.png")
    st.image(class_distribtion, caption="Distribution of Classes", use_column_width=False)
    
    st.divider()
    
    st.write("Example images from dataset")
    sample_images = Image.open("sample_images_screenshot.png")
    st.image(sample_images, caption = "Sample Images from Dataset")
    
    st.divider()
    
    st.write("Exam of Overlayed image used for processing in alternative model")
    overlayed_image = Image.open("Data/ImagesOverlay/malignant/malignant (1)_overlay.png")
    st.image(overlayed_image, caption="Sample Overlayed Image")
    
def metrics_tab():
    """
    Final metrics and results of model
    """
    st.title("Final Results and Metrics")
    
    selected_tab = st.selectbox("Select a Model", ["Original Model", "Mask Overlay Model"], key="selectbox")
    
    st.markdown(
        """
        <style>
            .stSelectbox {
                margin-top: 20px;
            }
            .st-5c8- {
                width: 50px; /* Set the desired width */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if selected_tab == "Original Model":
        st.write("Metrics (Macro)")
        
        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [0.7265, 0.7372, 0.7265, 0.7295]
        }
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

        st.write("Metrics by Class")
        metrics_class = {
            "Precision": [0.83, 0.60, 0.67],
            "Recall": [0.74, 0.66, 0.77],
            "F1 Score": [0.78, 0.62, 0.72]
        }
        class_df = pd.DataFrame(metrics_class, index=["Benign", "Malignant", "Normal"])
        st.table(class_df)
        
        # Confusion Matrix
        st.write("Confusion Matrix")
        confusion_matrix = "cm_original.png"
        cm_image = Image.open(confusion_matrix)
        st.image(cm_image, caption="Confusion Matrix", width=600)
        
        # Training loss
        st.write("Test Loss v. Validation Loss")
        loss_graph = "loss_original.png"
        loss_image = Image.open(loss_graph)
        st.image(loss_image, caption="Test v. Validation Loss", width=600)
        
        
    elif selected_tab == "Mask Overlay Model":
        st.write("Metrics (Macro)")
        
        metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [0.9786, 0.9786, 0.9786, 0.9786]
        }
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

        st.write("Metrics by Class")
        metrics_class = {
            "Precision": [0.98, 0.97, 1.00],
            "Recall": [0.99, 0.95, 1.00],
            'F1 Score': [0.98, 0.96, 1.00]
        }
        class_df = pd.DataFrame(metrics_class, index=["Benign", "Malignant", "Normal"])
        st.table(class_df)
        
        # Confusion Matrix
        st.write("Confusion Matrix")
        confusion_matrix = "cm_overlay.png"
        cm_image = Image.open(confusion_matrix)
        st.image(cm_image, caption="Confusion Matrix", width=600)
        
        # Training loss
        st.write("Test Loss v. Validation Loss")
        loss_graph = "loss_mask.png"
        loss_image = Image.open(loss_graph)
        st.image(loss_image, caption="Test v. Validation Loss", width=600)
    
def demo_tab():
    """
    Allow for a demo file to upload for the model to predict on
    """
    st.title("Demo: Predict on an Example image")
    
    selected_tab = st.selectbox("Select a Model", ["Original Model", "Mask Overlay Model"], key="selectbox")
    
    if selected_tab == "Original Model":
        # Upload an image for prediction
        image_upload = st.file_uploader("Select an image. . . ", type="png")
        
        if image_upload:
            image = Image.open(image_upload)
            st.image(image, caption="Uploaded Image to predict", use_column_width=False)
            
            # Placeholder prediction for now
            prediction = predict("trained_model_original.pt", image)
            
            st.write(f"Class prediction: {prediction}")
    
    if selected_tab == "Mask Overlay Model":
        # Upload an image for prediction
        image_upload = st.file_uploader("Select an image. . . ", type="png")
        
        if image_upload:
            image = Image.open(image_upload)
            st.image(image, caption="Uploaded Image to predict", use_column_width=False)
            
            # Placeholder prediction for now
            prediction = predict("trained_model.pt", image)
            
            st.write(f"Class prediction: {prediction}")
        
def main():
    st.set_page_config(page_title = "Breat Cancer Classification", layout='wide')
    
    st.markdown(
    """
    <style>
        body {
            background-color: #f0f0f0; /* Light pink background color */
        }
        h1 {
            color: #ff3399; /* Pink text color for headers */
        }
        p {
            color: #ff0066; /* Pink text color for paragraphs */
        }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    # Creating tabs
    tabs = ["Project Overview", "Results", "Demo"]
    current_tab = st.sidebar.radio("Select Tab", tabs)
    
    if current_tab == "Project Overview":
        overview_tab()
    elif current_tab == "Results":
        metrics_tab()
    elif current_tab == "Demo":
        demo_tab()

if __name__ == "__main__":
    main()
    
# My code from Code.py
# Note: Much of this code was collaborated on between Alejandra and myself, with both of us making edits where necessary
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

num_epochs = 20

def model_definition():
    # Defining Model
    model = models.resnet101(weights=True)
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

def train_model(train_loader, validation_loader, num_epochs, save_on=True):
    # Getting model
    model, optimizer, criterion = model_definition()
    f1_best = 0

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
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        # Validation phase for the original dataset
        model.eval()
        val_predictions = []
        val_labels = []

        test_loss = 0
        steps_test = 0

        with torch.no_grad():

            # with tqdm(total=len(validation_loader), desc="Epoch {}".format(epoch)) as pbar:

            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                # loss = criterion(predicted, labels)
                # test_loss += loss.item()
                # steps_test += 1

                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                # pbar.update(1)
                # pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

        # Calculate metrics for the original dataset
        accuracy = accuracy_score(val_labels, val_predictions)
        precision = precision_score(val_labels, val_predictions, average='weighted', zero_division=1)
        recall = recall_score(val_labels, val_predictions, average='weighted')
        f1 = f1_score(val_labels, val_predictions, average='weighted')

        print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        if save_on == True and f1 > f1_best:
            torch.save(model, "trained_model_original.pt")
            f1_best = f1

            print("The model has been saved and updated!")

    return model
#

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

acc = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred, average='weighted', zero_division=1)
recall = recall_score(test_labels, test_pred, average='weighted')
f1 = f1_score(test_labels, test_pred, average='weighted')

print('-' * 100)
print("METRICS ON TEST DATASET")
print(classification_report(test_labels, test_pred))
print()
print("CONFUSION MATRIX")
print(confusion_matrix(test_labels, test_pred))
print()
print(f"Accuracy: {acc.__round__(4)}, Precision: {precision.__round__(4)}, Recall: {recall.__round__(4)}, F1: {f1.__round__(4)}")

# ## Applying model to augment data
model = train_model(train_loader_o, validation_loader_o, num_epochs)
# #
## Printing Metrics for the test data
test_pred_o = []
test_labels_o = []

# Getting predictions
with torch.no_grad():
    for images, labels in test_loader_o:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        test_pred_o.extend(predicted.cpu().numpy())
        test_labels_o.extend(labels.cpu().numpy())

# Metrics on Overlayed Images
acc = accuracy_score(test_labels_o, test_pred_o)
precision = precision_score(test_labels_o, test_pred_o, average='weighted', zero_division=1)
recall = recall_score(test_labels_o, test_pred_o, average='weighted')
f1 = f1_score(test_labels_o, test_pred_o, average='weighted')

print('-' * 100)
print("METRICS ON TEST DATASET")
print(classification_report(test_labels_o, test_pred_o))
print()
print("CONFUSION MATRIX")
print(confusion_matrix(test_labels_o, test_pred_o))
print()
print(f"Accuracy: {acc.__round__(4)}, Precision: {precision.__round__(4)}, Recall: {recall.__round__(4)}, F1: {f1.__round__(4)}")

image_ids, image_labels = test_dataset_o.get_ids_and_labels()

test_image_df = pd.DataFrame({'image_id': image_ids, 'labels': image_labels})
test_image_df.to_csv('test_image_df.csv')

# Toolbox Code
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

# Gaussian Noise
class GaussianNoise(object):
    def __init__(self, std=0.001):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        noisy_tensor = tensor + noise

        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

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

def predict(model_path, image):
    """
    Makes a prediction of an imputed image using the trained model
    
    input:  model_path: path for saved model; 
            image: previously loaded image to be predicted
    ouput:  predicted_labl: Prediction from the model
    """
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)
    # model.to(device)
    
    # Get image and preprocess
    transform = transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Choose the desired size
            transforms.ToTensor(),
            # Add any other transformations you need
    ])
    # image = image.convert("RBG")
    img = transform(image)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(device)
    
    # Making prediction
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    
    _, predicted_idx = torch.max(output, 1)
    
    class_labels = ["benign", "malignant", "normal"]
    predicted_label = class_labels[predicted_idx.item()]
    
    return predicted_label