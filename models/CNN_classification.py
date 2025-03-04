import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import json

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def load_dataset(root_path, img_size=(62, 47)):
    """
    Load images from LFW dataset directories where each directory name is a class label.
    Returns images, labels, and a mapping from label indices to class names.
    """
    images = []
    labels = []
    class_names = []

    # Get sorted list of directories to ensure consistent class indices
    person_folders = sorted([d for d in os.listdir(root_path)
                             if os.path.isdir(os.path.join(root_path, d))])

    # Create mapping from class index to class name
    class_to_idx = {name: i for i, name in enumerate(person_folders)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    for person_folder in person_folders:
        person_path = os.path.join(root_path, person_folder)
        class_names.append(person_folder)

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            if not os.path.isfile(img_path) or not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize to [0,1]

                images.append(img_array)
                labels.append(class_to_idx[person_folder])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels), idx_to_class


def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model for facial classification that works with 62x47 images.
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(dataset_path, img_size=(62, 47), batch_size=32, epochs=20, validation_split=0.2):
    """
    Train the CNN model and return the trained model and class mapping.
    """
    # Load dataset
    print("Loading dataset...")
    X, y, idx_to_class = load_dataset(dataset_path, img_size)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, stratify=y, random_state=42)

    print(f"Training with {len(X_train)} images, validating with {len(X_val)} images")
    print(f"Number of classes: {len(idx_to_class)}")
    print(f"Image shape: {X_train[0].shape}")

    # Data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Build the model
    input_shape = X_train[0].shape
    model = build_cnn_model(input_shape, len(idx_to_class))

    # Model summary
    model.summary()

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Learning rate reduction on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"\nValidation accuracy: {test_acc:.4f}")

    # Plot training history
    plot_training_history(history)

    return model, idx_to_class, (X_val, y_val)


def plot_training_history(history):
    """
    Plot training and validation accuracy/loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def process_test_image(image_path, target_size=(62, 47)):
    """
    Process a test image for classification:
    - Resize to target dimensions (62x47)
    - Normalize pixel values to 0.0-1.0 range
    - Handle different image formats

    Returns processed image ready for model input
    """
    try:
        # Load image using OpenCV
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        else:
            # If already a numpy array
            img = image_path
            if len(img.shape) == 3 and img.shape[2] == 4:  # Handle RGBA
                img = img[:, :, :3]

        # Resize image to 62x47 as specified for the LFW dataset
        img = cv2.resize(img, target_size)

        # Normalize pixel values to [0,1]
        img = img.astype(np.float32) / 255.0

        return img

    except Exception as e:
        print(f"Error processing test image: {e}")
        return None


def classify_new_image(model, idx_to_class, image_path, img_size=(62, 47)):
    """
    Classify a new image using the trained model.
    """
    try:
        # Process the image
        if isinstance(image_path, str):
            processed_img = process_test_image(image_path, img_size)
        else:
            # If it's already a numpy array
            processed_img = image_path if image_path.max() <= 1.0 else image_path / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(processed_img, axis=0)

        # Make prediction
        predictions = model.predict(img_batch)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]

        person_name = idx_to_class[predicted_class_idx]

        # Display results
        print(f"Predicted person: {person_name}")
        print(f"Confidence: {confidence:.4f}")

        # Show top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        print("\nTop 5 predictions:")
        for idx in top_5_indices:
            print(f"{idx_to_class[idx]}: {predictions[0][idx]:.4f}")

        # Display the image with prediction
        plt.figure(figsize=(4, 6))
        plt.imshow(processed_img)
        plt.title(f"Predicted: {person_name}\nConfidence: {confidence:.4f}")
        plt.axis('off')
        plt.show()

        return person_name, confidence, predictions[0]

    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, 0.0, None


def main():
    # Path to your dataset
    dataset_path = r"C:\Users\HP\Downloads\archive\lfw-funneled\lfw_funneled"

    # Train the model with the LFW specific dimensions (62x47)
    model, idx_to_class, val_data = train_model(
        dataset_path,
        img_size=(62, 47),
        batch_size=32,
        epochs=25
    )

    # Save the model and class mapping
    model.save('lfw_face_classification_model.h5')

    # Save class mapping to file
    with open('lfw_class_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f)

    print("Model and class mapping saved.")

    # Test the model on some validation images
    X_val, y_val = val_data
    num_test = min(5, len(X_val))
    test_indices = np.random.choice(len(X_val), num_test, replace=False)

    for idx in test_indices:
        test_img = X_val[idx]
        true_label = idx_to_class[y_val[idx]]
        print(f"\nTrue label: {true_label}")

        # Test classification
        classify_new_image(model, idx_to_class, test_img)


def load_saved_model(model_path='lfw_face_classification_model.h5',
                     mapping_path='lfw_class_mapping.json'):
    """
    Load a previously saved model and class mapping.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load class mapping
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
        # Convert string keys back to integers
        idx_to_class = {int(k): v for k, v in class_mapping.items()}

    return model, idx_to_class


if __name__ == "__main__":
    main()