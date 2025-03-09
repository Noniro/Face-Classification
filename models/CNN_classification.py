import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Verify TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# Import the utility functions - using sys.path to handle the imports
import sys

sys.path.append('.')  # Add current directory to path
from utils.data_loader import load_data
from utils.data_splitter import split_data

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
dataset_path = 'Data-peoplefaces'  # Path to people faces dataset

# Load the data using the provided utility
X, y, label_dict = load_data(dataset_path)
print(f"Loaded {len(X)} images with {len(label_dict)} classes")
print(f"Image shape: {X[0].shape}")

# Split the data into training and testing sets
X_train, X_test, y_train_cat, y_test_cat = split_data(X, y)
print(f"Training set: {X_train.shape}, {y_train_cat.shape}")
print(f"Testing set: {X_test.shape}, {y_test_cat.shape}")

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

datagen.fit(X_train)


# CNN Architecture for 184 class face recognition
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

# Build and compile the model
model = build_cnn_model(input_shape, num_classes)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Define callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
    ModelCheckpoint('best_face_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

# Train the model with data augmentation
batch_size = 32
epochs = 50

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")


# Plot the training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
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