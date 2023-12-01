from Toolbox import CancerDataset, EDA, get_balanced_augmentation_transform, balanced_augmentation
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
from tqdm import tqdm

#Loading dataset

# path = "/home/ubuntu/Final_Project/Data/"
path = "/home/ubuntu/final-project/6303_10_Final_Project_Group5/Data"
augment_transform = get_balanced_augmentation_transform(horizontal_flip=True,
                                                vertical_flip=True,
                                                rotation_angle=45,
                                                gaussian_noise=True,
                                                brightness_range=0.2,
                                                contrast_range=0.2,
                                                saturation_range=0.2)
cancer_dataset = CancerDataset(path)

train_dataset, validation_dataset, test_dataset = cancer_dataset.split_dataset(test_size=0.3, validation_size=0.2)

# Create DataLoader instances for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# #EDA prior augmentation - Commenting out for now
# eda = EDA(path)
# eda.plot_class_distribution()
# eda.plot_sample_images()


#Splitting data
train_dataset, val_dataset, test_dataset = cancer_dataset.split_dataset(test_size=0.3, validation_size=0.2, random_seed=42)
batch_size = 32
num_epochs = 20


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def model_definition():
    # Defining Model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    
    # Defining Criteria
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Consider using this scheduler
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)
    
    return model, optimizer, criterion
    
# Training loop
def train_model(train_loader, validation_loader, num_epochs, save_on=True):
    # Getting model
    model, optimizer, criterion = model_definition()
    f1_best = 0
    
    # Beggining Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        steps_train = 0
        
        with tqdm(total=len(train_loader), desc="Epoch {}".format(epoch+1)) as pbar:
        
            for images, labels in train_loader:
                # Augmenting image
                images = torch.stack([augment_transform(img) for img in images])
                
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
            torch.save(model.state_dict(), "trained_model.pt")
            f1_best = f1
            
            print("The model has been saved and updated!")
        
        
    return model


## Applying model to augment data
model = train_model(train_loader, validation_loader, num_epochs)

## Printing Metrics for the test data
test_pred = []
test_labels = []
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Getting predictions
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        test_pred.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        
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
print(f"Accuracy: {acc.__round__(4)}, Precision: {precision.__round__(4)}, Recall: {recall.__round__(4)}, F1: {f1.__round__(4)}")