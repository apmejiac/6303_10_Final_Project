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

# path = "/home/ubuntu/DL_Proj/Data/"
path = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data"
output_path_mask = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data/ImagesMask/"
output_path_without_mask = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data/ImagesWithoutMask/"
output_path_overlay = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data/ImagesOverlay/"

dataset_creator = DatasetCreator(path)

#Creates dataset on virtual machine (comment it out after having them on virtual machine)
# dataset_creator.create_dataset_mask_folder(output_path_mask)
# dataset_creator.create_images_without_mask_folder(output_path_without_mask)
# overlay_images_with_masks(output_path_mask, output_path_without_mask, output_path_overlay)

#========================================================================================================

### Complete dataset EDA
# eda_C = EDA(path)
# eda_C.plot_class_distribution(custom_title='Class distribution raw dataset')
# eda_C.plot_sample_images()


#========================================================================================================
#Dataset with only original images
cancer_dataset_original = CancerDataset(output_path_without_mask)
train_dataset, validation_dataset, test_dataset = cancer_dataset_original.split_dataset(test_size=0.3, validation_size=0.2)
#
# # Create DataLoader instances for training, validation, and test sets
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

# eda = EDA(output_path_without_mask)
# eda.plot_class_distribution(custom_title='Class distribution only images no mask')
# eda.plot_sample_images()


# #Considering data augmentation for training dataset
minority_classes = ['malignant', 'normal']

# Apply augmentation to the minority classes in the dataset
augmented_dataset = apply_augmentation_to_dataset(train_dataset, horizontal_flip=True,
                                                  vertical_flip=True, rotation_angle=90,
                                                  brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                                                  saturation_range=(0.8, 1.2), normalize=True, classes_to_augment=minority_classes)
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




#formatted confusion matrix
cm = confusion_matrix(test_labels, test_pred)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix Original data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#
#
#
#

#========================================================================================================

#Dataset overlayed images (original + masks)
cancer_dataset_overlayed= CancerDataset(output_path_overlay)
train_dataset_o, validation_dataset_o, test_dataset_o = cancer_dataset_overlayed.split_dataset(test_size=0.3, validation_size=0.2)
image_ids, image_labels = test_dataset_o.get_ids_and_labels()

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

def train_model(train_loader_o, validation_loader_o, num_epochs, save_on=True):
    # Getting model
    model, optimizer, criterion = model_definition()
    f1_best = 0

    # Beggining Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps_train = 0

        with tqdm(total=len(train_loader_o), desc="Epoch {}".format(epoch + 1)) as pbar:

            for images, labels in train_loader_o:
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

            average_loss = total_loss / len(train_loader_o)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}')

        # Validation phase for the original dataset
        model.eval()
        val_predictions = []
        val_labels = []

        test_loss = 0
        steps_test = 0

        with torch.no_grad():

            # with tqdm(total=len(validation_loader), desc="Epoch {}".format(epoch)) as pbar:

            for images, labels in validation_loader_o:
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

        print(
            f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        if save_on == True and f1 > f1_best:
            torch.save(model, "trained_model.pt")
            f1_best = f1

            print("The model has been saved and updated!")

    return model
#
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



#formatted confusion matrix
cm = confusion_matrix(test_labels_o, test_pred_o)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.title('Confusion Matrix Overlayed data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate performance on regular images with overlayed model
# Need to make equivalent dataloader with regular images
# for batch in test_loader_o:
#     for item in batch:
#         print(item)

# Get metrics on regular images

# cancer_dataset_overlayed= CancerDataset(output_path_without_mask)
# train_dataset_o, validation_dataset_o, test_dataset_o = cancer_dataset_overlayed.split_dataset(test_size=0.3, validation_size=0.2)
# image_ids, image_labels = test_dataset_o.get_ids_and_labels()

# test_image_df = pd.DataFrame({'image_id': image_ids, 'labels': image_labels})
# print(test_image_df.head())

# predictions = []
# for index, row in test_image_df.iterrows():
#     image = Image.open(row['image_id'])
#     pred = predict('trained_model.pt', image)
#     predictions.append(pred)

# test_image_df["prediction"] = predictions

# print(classification_report(test_image_df['labels'], test_image_df['prediction']))
# print(confusion_matrix(test_image_df['labels'], test_image_df['prediction']))