from Toolbox import CancerDataset,EDA,get_balanced_augmentation_transform,balanced_augmentation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch

#Loading dataset

# path = "/home/ubuntu/Final_Project/Data/"
path = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data"
cancer_dataset = CancerDataset(path)

train_dataset, validation_dataset, test_dataset = cancer_dataset.split_dataset(test_size=0.3, validation_size=0.2)

# Create DataLoader instances for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#EDA prior augmentation
eda = EDA(path)
eda.plot_class_distribution()
eda.plot_sample_images()


#Splitting data
train_dataset, val_dataset, test_dataset = cancer_dataset.split_dataset(test_size=0.3, validation_size=0.2, random_seed=42)
batch_size = 32


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

pretrained_resnet = models.resnet18(pretrained=True)
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_resnet.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    pretrained_resnet.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = pretrained_resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss:.4f}')

    # Validation phase for the original dataset
    pretrained_resnet.eval()
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for images, labels in validation_loader:
            outputs = pretrained_resnet(images)
            _, predicted = torch.max(outputs, 1)

            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    # Calculate metrics for the original dataset
    accuracy = accuracy_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions, average='weighted', zero_division=1)
    recall = recall_score(val_labels, val_predictions, average='weighted')
    f1 = f1_score(val_labels, val_predictions, average='weighted')

    print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')


## Applying model to augment data

## Printing Metrics for the test data
test_pred = []
test_labels = []
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Getting predictions
with torch.no_grad():
    for images, labels in test_loader:
        outputs = pretrained_resnet(images)
        _, predicted = torch.max(outputs, 1)
        
        test_pred.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cup().numpy())
        
acc = accuracy_score(test_labels, test_pred)
precision = precision_score(test_labels, test_pred, average='weighted', zero_division=1)
recall = recall_score(test_labels, test_pred, average='weighted')
f1 = f1_score(test_labels, test_pred, average='weighted')

print('-'*100)
print("METRICS ON TEST DATASET")
print(classification_report(test_labels, test_pred))
print()
print("CONFUSION MATRIX")
print(confusion_matrix(test_labels, test_pred))
print()
print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")