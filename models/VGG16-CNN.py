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

# Split the data into training and testing sets
X_train, X_test, y_train_cat, y_test_cat = split_data(X, y)
print(f"Training set: {X_train.shape}, {y_train_cat.shape}")
print(f"Testing set: {X_test.shape}, {y_test_cat.shape}")

# Convert one-hot encoded labels back to class indices
y_train = np.argmax(y_train_cat, axis=1)
y_test = np.argmax(y_test_cat, axis=1)


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
num_classes = len(label_dict)
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
            torch.save(model.state_dict(), 'best_pytorch_face_model.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')

    # Load best model
    model.load_state_dict(torch.load('best_pytorch_face_model.pth'))
    return model, history


# Train the model
print("Starting training...")
trained_model, history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)


# Plot training history
def plot_training_history(history):
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
    plt.savefig('pytorch_training_history.png')
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
plot_training_history(fine_tune_history)

# Evaluate fine-tuned model
print("Evaluating fine-tuned model...")
fine_tuned_accuracy = evaluate_model(fine_tuned_model, test_loader)

# Save the fine-tuned model
torch.save(fine_tuned_model.state_dict(), 'pytorch_face_finetuned_model.pth')
print("Fine-tuned model saved as 'pytorch_face_finetuned_model.pth'")


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