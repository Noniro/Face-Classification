import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import pickle
from collections import Counter

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up one level to Face-Classification
sys.path.append(parent_dir)  # Add parent directory to path

# Import the utility functions
try:
    from utils.data_loader import load_data
    from utils.data_splitter import split_data

    print("Successfully imported from utils package")
except ImportError:
    # Fallback to direct module import
    utils_dir = os.path.join(parent_dir, 'utils')
    sys.path.append(utils_dir)
    import data_loader
    import data_splitter

    load_data = data_loader.load_data
    split_data = data_splitter.split_data
    print("Using fallback direct module imports")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define paths
dataset_path = os.path.join(parent_dir, 'Data-peopleFaces')  # Path to people faces dataset

# Load the data using the provided utility
print(f"Loading data from: {dataset_path}")
X, y, label_dict = load_data(dataset_path, img_size=(224, 224))  # VGG16 requires 224x224 images
print(f"Loaded {len(X)} images with {len(label_dict)} classes")
print(f"Image shape: {X[0].shape}")

# Save the original label dictionary for reference
with open('label_dict_original.pkl', 'wb') as f:
    pickle.dump(label_dict, f)
print("Original label dictionary saved as 'label_dict_original.pkl'")

# Filter people with at least 15 images
# Check if labels are already indices or one-hot encoded
if len(y.shape) == 1 or y.shape[1] == 1:
    # Labels are already indices
    y_indices = y.astype(int)
    if len(y_indices.shape) > 1:
        y_indices = y_indices.flatten()
    print("Labels are already in index format")
else:
    # Labels are one-hot encoded
    y_indices = np.argmax(y, axis=1)
    print("Converted one-hot encoded labels to indices")

# Count images per class
class_counts = Counter(y_indices)
print(f"Class distribution before filtering: {class_counts.most_common(10)}")  # Show only top 10 to avoid clutter

# Identify classes with at least 15 images
minimum_images = 15
valid_classes = [class_id for class_id, count in class_counts.items() if count >= minimum_images]
print(f"Found {len(valid_classes)} classes with at least 15 images")

# Create a mapping from old class indices to new class indices
old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_classes)}

# Create a new label dictionary with only valid classes
new_label_dict = {new_idx: label_dict[old_idx] for old_idx, new_idx in old_to_new_idx.items()}
print(f"New label dictionary has {len(new_label_dict)} classes")

# Filter the data to include only images from valid classes
valid_indices = [i for i, label_idx in enumerate(y_indices) if label_idx in valid_classes]
X_filtered = X[valid_indices]
y_indices_filtered = [old_to_new_idx[y_indices[i]] for i in valid_indices]

# Convert the filtered indices back to one-hot encoding
num_classes = len(valid_classes)
if len(y.shape) == 1 or y.shape[1] == 1:
    # Keep as indices for now, convert later if needed
    y_filtered = np.array(y_indices_filtered)
    print(f"Keeping filtered labels as indices, shape: {y_filtered.shape}")
else:
    # Convert back to one-hot encoding
    y_filtered = np.zeros((len(y_indices_filtered), num_classes))
    for i, label_idx in enumerate(y_indices_filtered):
        y_filtered[i, label_idx] = 1
    print(f"Converted filtered labels to one-hot encoding, shape: {y_filtered.shape}")

print(f"Filtered dataset: {X_filtered.shape}, {y_filtered.shape}")

# Save the new label dictionary
with open('label_dict_filtered.pkl', 'wb') as f:
    pickle.dump(new_label_dict, f)
print("Filtered label dictionary saved as 'label_dict_filtered.pkl'")

# Split the filtered data into training and testing sets
X_train, X_test, y_train_cat, y_test_cat = split_data(X_filtered, y_filtered)
print(f"Training set: {X_train.shape}, {y_train_cat.shape}")
print(f"Testing set: {X_test.shape}, {y_test_cat.shape}")

# Convert labels to class indices if they're one-hot encoded
if len(y_train_cat.shape) > 1 and y_train_cat.shape[1] > 1:
    y_train = np.argmax(y_train_cat, axis=1)
    y_test = np.argmax(y_test_cat, axis=1)
    print("Converted one-hot encoded training/testing labels to indices")
else:
    # Already in index format
    y_train = y_train_cat
    y_test = y_test_cat
    print("Training/testing labels already in index format")


# Create PyTorch Dataset
class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# Create datasets
train_dataset = FaceDataset(X_train, y_train, train_transform)
test_dataset = FaceDataset(X_test, y_test, test_transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# Define the model
def build_model(num_classes):
    # Load pre-trained VGG16 model
    model = models.vgg16(weights='DEFAULT')

    # Freeze feature layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify classifier for our number of classes
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, num_classes)
    )

    return model


# Instantiate the model
model = build_model(num_classes)
model = model.to(device)

# Print model summary
print(model)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=1e-6)


# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_accuracy = 0.0
    model_filename = f'best_vgg_model_{num_classes}_classes.pth'
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        # No gradients needed for validation
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Statistics
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})

        val_epoch_loss = val_running_loss / len(test_dataset)
        val_epoch_acc = val_running_corrects.double() / len(test_dataset)

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # Update learning rate scheduler
        scheduler.step(val_epoch_loss)

        # Save best model
        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            torch.save(model.state_dict(), model_filename)
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')

    # Load best model
    model.load_state_dict(torch.load(model_filename))
    return model, history


# Train the model
print("Starting training...")
trained_model, history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)


# Plot training history
def plot_training_history(history, filename_prefix="filtered"):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_training_history.png')
    # Skip plt.show() to avoid PyCharm errors


# Plot the training history
plot_training_history(history)


# Evaluate final model
def evaluate_model(model, test_loader):
    model.eval()
    corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            corrects += (preds == labels).sum().item()

    accuracy = 100 * corrects / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# Evaluate the model
print("Evaluating final model...")
final_accuracy = evaluate_model(trained_model, test_loader)

# Fine-tuning - unfreeze some VGG layers and train with lower learning rate
print("\nFine-tuning the model by unfreezing some VGG16 layers...")

# Unfreeze the last few convolutional layers
for param in trained_model.features[-4:].parameters():
    param.requires_grad = True

# Use a much lower learning rate for fine-tuning
optimizer = optim.Adam([
    {'params': trained_model.features[-4:].parameters(), 'lr': 1e-5},
    {'params': trained_model.classifier.parameters(), 'lr': 1e-4}
])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, min_lr=1e-7)

# Fine-tune for a few more epochs
fine_tuned_model, fine_tune_history = train_model(
    trained_model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=20
)

# Plot fine-tuning history
plot_training_history(fine_tune_history, filename_prefix="filtered_finetuned")

# Evaluate fine-tuned model
print("Evaluating fine-tuned model...")
fine_tuned_accuracy = evaluate_model(fine_tuned_model, test_loader)

# Save the fine-tuned model
finetuned_filename = f'finetuned_vgg_model_{num_classes}_classes.pth'
torch.save(fine_tuned_model.state_dict(), finetuned_filename)
print(f"Fine-tuned model saved as '{finetuned_filename}'")

# Function to predict a person's identity
def predict_person(model, image_path, label_dict):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Apply transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][preds[0]].item()

    # Get the person's name
    predicted_class = preds[0].item()
    person_name = label_dict[predicted_class]

    return person_name, confidence


print("\nTraining completed! Check the saved models and training history plots.")
print(f"Filtered model trained on {num_classes} classes (people with "+minimum_images+"+ images)")
print(f"Models saved as:")
print(f"- best_vgg_model_{num_classes}_classes.pth")
print(f"- finetuned_vgg_model_{num_classes}_classes.pth")
print(f"- label_dict_filtered.pkl")