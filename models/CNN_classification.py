import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from collections import Counter

# Alternative imports that work better with PyCharm
keras = tf.keras
Sequential = tf.keras.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
Adam = tf.keras.optimizers.Adam
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Verify TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

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
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
dataset_path = os.path.join(parent_dir, 'Data-peopleFaces')  # Path to people faces dataset


# Function to filter people with minimum number of photos
def filter_people_with_min_photos(dataset_path, min_photos=15):
    """
    Filters out people who don't have the minimum required number of photos.

    Args:
        dataset_path: Path to the dataset directory
        min_photos: Minimum number of photos required per person

    Returns:
        filtered_paths: List of file paths for people who meet the minimum photo requirement
        filtered_labels: List of corresponding labels
        filtered_label_dict: Dictionary mapping indices to person names
    """
    print(f"Filtering people with at least {min_photos} photos...")

    # Get all subdirectories (person folders)
    person_folders = [f for f in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, f))]

    filtered_paths = []
    filtered_labels = []
    person_counts = {}

    # First pass: count photos per person
    for i, person in enumerate(person_folders):
        person_dir = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        person_counts[person] = len(image_files)

    # Filter people with enough photos
    qualified_people = [person for person, count in person_counts.items()
                        if count >= min_photos]

    if not qualified_people:
        print(f"Warning: No people have {min_photos} or more photos. Using all available data.")
        return load_data(dataset_path)

    # Second pass: collect paths and labels for qualified people
    filtered_label_dict = {}
    for i, person in enumerate(qualified_people):
        person_dir = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in image_files:
            img_path = os.path.join(person_dir, img_file)
            filtered_paths.append(img_path)
            filtered_labels.append(i)

        filtered_label_dict[i] = person

    print(f"Found {len(qualified_people)} people with at least {min_photos} photos.")
    print(f"Total images after filtering: {len(filtered_paths)}")

    # Load and process the filtered images
    X = []
    y = np.array(filtered_labels)

    for img_path in filtered_paths:
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (100, 100))  # Resize to a standard size
        img = img.astype('float32') / 255.0  # Normalize to [0,1]
        X.append(img)

    X = np.array(X)

    return X, y, filtered_label_dict


# Load the filtered data
MIN_PHOTOS_REQUIRED = 15  # Define minimum photos required per person
X, y, label_dict = filter_people_with_min_photos(dataset_path, MIN_PHOTOS_REQUIRED)

print(f"Loaded {len(X)} images with {len(label_dict)} classes")
print(f"Image shape: {X[0].shape}")

# Add debugging information about dataset
print("\nDataset Statistics:")
print(f"Total images: {len(X)}")
print(f"Number of classes: {len(label_dict)}")

# Check for potential issues in the dataset
print("\nChecking for potential issues:")
print(f"Image shapes consistent: {all(img.shape == X[0].shape for img in X)}")
print(f"Value ranges: min={np.min(X)}, max={np.max(X)}")

# Print class distribution
class_counts = np.bincount(y)
print("\nClass distribution (samples per person):")
min_samples = np.min(class_counts)
max_samples = np.max(class_counts)
avg_samples = np.mean(class_counts)
print(f"Min: {min_samples}, Max: {max_samples}, Avg: {avg_samples:.1f}")

# Split the data into training and testing sets
X_train, X_test, y_train_cat, y_test_cat = split_data(X, y)
print(f"Training set: {X_train.shape}, {y_train_cat.shape}")
print(f"Testing set: {X_test.shape}, {y_test_cat.shape}")

# Simplified data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Subtle rotation
    width_shift_range=0.1,  # Subtle shift
    height_shift_range=0.1,  # Subtle shift
    horizontal_flip=True,  # Horizontal flip is safe
    zoom_range=0.1,  # Subtle zoom
    # Note: no rescale parameter as data is already normalized
    fill_mode='nearest'
)

# Fit the data generator on training data
datagen.fit(X_train)


# Function to visualize augmented images
def visualize_augmentation(datagen, X_samples, y_samples, label_dict, num_images=5):
    # Select a few images for visualization
    indices = np.random.choice(range(len(X_samples)), min(num_images, len(X_samples)), replace=False)
    X_sample = X_samples[indices]
    y_sample = y_samples[indices]

    # Get a batch of augmented images
    aug_gen = datagen.flow(X_sample, y_sample, batch_size=num_images)
    X_batch, y_batch = next(aug_gen)

    # Create the plot
    plt.figure(figsize=(15, 2 * num_images))

    # Plot original and augmented images side by side
    for i in range(num_images):
        # Original image
        plt.subplot(num_images, 2, 2 * i + 1)
        plt.imshow(X_sample[i])
        plt.title(f"Original: {label_dict[y_sample[i]]}")
        plt.axis('off')

        # Augmented image
        plt.subplot(num_images, 2, 2 * i + 2)
        plt.imshow(X_batch[i])
        plt.title(f"Augmented: {label_dict[y_batch[i]]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('augmented_samples.png')
    plt.show()


# Visualize some augmented images to verify augmentation
print("\nVisualizing augmented images:")
visualize_augmentation(datagen, X_train, np.argmax(y_train_cat, axis=1), label_dict)


# CNN Architecture for face recognition
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


# Get the input shape from the training data
input_shape = X_train[0].shape
num_classes = y_train_cat.shape[1]

# Build and compile the model - using the original architecture
model = build_cnn_model(input_shape, num_classes)
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Original learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Define callbacks for training - keep early stopping but with more patience
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6, mode='max'),
    ModelCheckpoint('best_face_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

# Train the model with data augmentation
batch_size = 32  # Original batch size
epochs = 50  # Moderate number of epochs

# Generate augmented samples - used fixed steps to ensure all classes are included
steps_per_epoch = len(X_train) // batch_size

# Train the model with augmented data
print("\nStarting model training with augmented data:")
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_test, y_test_cat),  # Use the test set directly for validation
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")


# Plot the training history
def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# Plot the training history
plot_training_history(history)


# Function to predict a person's identity
def predict_person(model, image, label_dict):
    # Ensure image is in the right format
    if image.shape != X_train[0].shape:
        image = cv2.resize(image, (X_train[0].shape[1], X_train[0].shape[0]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32') / 255.0

    # Reshape for model input
    image = np.expand_dims(image, axis=0)

    # Get prediction
    prediction = model.predict(image)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Get the person's name
    person_name = label_dict[predicted_class]

    return person_name, confidence


# Save the trained model
model.save('face_classification_model.h5')
print("Model saved as 'face_classification_model.h5'")

# Example of how to use the model for prediction
# You would typically load a new image here
# sample_image = cv2.imread('path_to_new_image.jpg')
# person_name, confidence = predict_person(model, sample_image, label_dict)
# print(f"Predicted person: {person_name} with confidence: {confidence:.2f}")

# Print final model statistics
print("\nFinal Model Statistics:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Final test accuracy: {test_accuracy:.2%}")
print(f"Parameters used:")
print(f" - Batch size: {batch_size}")
print(f" - Learning rate: 0.001")
print(f" - Augmentation: rotation={10}°, width/height shift={0.1}, zoom={0.1}, horizontal_flip=True")