import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import pickle
import json


def load_dataset(root_path, img_size=(62, 47)):
    """
    Load images and convert them to feature vectors.
    Returns feature vectors, labels, and a mapping from label indices to class names.
    """
    images = []
    labels = []

    # Get sorted list of directories to ensure consistent class indices
    person_folders = sorted([d for d in os.listdir(root_path)
                             if os.path.isdir(os.path.join(root_path, d))])

    # Create mapping from class index to class name
    class_to_idx = {name: i for i, name in enumerate(person_folders)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    print(f"Loading images from {len(person_folders)} people...")

    for person_folder in person_folders:
        person_path = os.path.join(root_path, person_folder)

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            if not os.path.isfile(img_path) or not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0  # Normalize to [0,1]

                # Flatten the image into a feature vector
                feature_vector = img_array.flatten()

                images.append(feature_vector)
                labels.append(class_to_idx[person_folder])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels), idx_to_class


def extract_features(images, n_components=150):
    """
    Extract features using PCA to reduce dimensionality.
    Decision trees work better with reduced feature space.
    """
    print(f"Original feature dimensions: {images.shape}")

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components, whiten=True)
    features = pca.fit_transform(images)

    print(f"Reduced feature dimensions with PCA: {features.shape}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")

    return features, pca


def train_decision_tree(features, labels, idx_to_class):
    """
    Train a decision tree classifier on the extracted features.
    """
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training with {X_train.shape[0]} images, testing with {X_test.shape[0]} images")

    # Grid search for optimal hyperparameters
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    print("Finding optimal hyperparameters...")
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")

    # Get the best model
    dt_classifier = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize feature importance
    feature_importance = dt_classifier.feature_importances_
    # Get indices of top 10 most important features
    top_indices = np.argsort(feature_importance)[-10:]

    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importance')
    plt.barh(range(10), feature_importance[top_indices], align='center')
    plt.yticks(range(10), [f'Feature {i}' for i in top_indices])
    plt.xlabel('Importance')
    plt.savefig('dt_feature_importance.png')

    return dt_classifier, (X_test, y_test)


def process_test_image(image_path, pca, img_size=(62, 47)):
    """
    Process a test image for classification with a decision tree.
    """
    try:
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path

        # Resize image
        img = cv2.resize(img, img_size)

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        # Flatten the image into a feature vector
        feature_vector = img.flatten().reshape(1, -1)

        # Apply the same PCA transformation
        feature_vector = pca.transform(feature_vector)

        return feature_vector

    except Exception as e:
        print(f"Error processing test image: {e}")
        return None


def classify_with_decision_tree(dt_classifier, pca, idx_to_class, image_path, img_size=(62, 47)):
    """
    Classify an image using the trained decision tree.
    """
    try:
        # Process the image
        if isinstance(image_path, str):
            img_raw = cv2.imread(image_path)
            img_display = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            feature_vector = process_test_image(image_path, pca, img_size)
        else:
            img_display = image_path if image_path.max() <= 1.0 else image_path / 255.0
            feature_vector = process_test_image(image_path, pca, img_size)

        # Make prediction
        prediction = dt_classifier.predict(feature_vector)
        predicted_class_idx = prediction[0]

        # Get probability estimates if available
        if hasattr(dt_classifier, 'predict_proba'):
            probabilities = dt_classifier.predict_proba(feature_vector)
            confidence = probabilities[0][predicted_class_idx]
        else:
            confidence = 1.0  # Default if not available

        person_name = idx_to_class[predicted_class_idx]

        # Display results
        print(f"Predicted person: {person_name}")
        print(f"Confidence: {confidence:.4f}")

        # Display the image with prediction
        plt.figure(figsize=(4, 6))
        plt.imshow(img_display)
        plt.title(f"Predicted: {person_name}\nConfidence: {confidence:.4f}")
        plt.axis('off')
        plt.show()

        return person_name, confidence

    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, 0.0


def main():
    # Path to your dataset
    dataset_path = r"C:\Users\HP\Downloads\archive\lfw-funneled\lfw_funneled"

    # Load dataset
    print("Loading dataset...")
    X, y, idx_to_class = load_dataset(dataset_path)

    # Extract features using PCA
    print("Extracting features...")
    features, pca = extract_features(X)

    # Train decision tree
    print("Training decision tree classifier...")
    dt_classifier, test_data = train_decision_tree(features, y, idx_to_class)

    # Save the model and related objects
    with open('dt_face_classifier.pkl', 'wb') as f:
        pickle.dump(dt_classifier, f)

    with open('dt_pca.pkl', 'wb') as f:
        pickle.dump(pca, f)

    with open('dt_class_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f)

    print("Model, PCA transformer, and class mapping saved.")

    # Test on some test images
    X_test, y_test = test_data
    num_test = min(5, len(X_test))
    test_indices = np.random.choice(len(X_test), num_test, replace=False)

    for idx in test_indices:
        # Convert PCA features back to approximate image for display
        test_img_pca = X_test[idx].reshape(1, -1)
        test_img_approx = pca.inverse_transform(test_img_pca).reshape(47, 62, 3)
        test_img_approx = np.clip(test_img_approx, 0, 1)  # Ensure values are in valid range

        true_label = idx_to_class[y_test[idx]]
        print(f"\nTrue label: {true_label}")

        # Test classification
        classify_with_decision_tree(dt_classifier, pca, idx_to_class, test_img_approx)


def load_saved_dt_model(model_path='dt_face_classifier.pkl',
                        pca_path='dt_pca.pkl',
                        mapping_path='dt_class_mapping.json'):
    """
    Load a previously saved decision tree model and related objects.
    """
    # Load model
    with open(model_path, 'rb') as f:
        dt_classifier = pickle.load(f)

    # Load PCA transformer
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    # Load class mapping
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
        # Convert string keys back to integers
        idx_to_class = {int(k): v for k, v in class_mapping.items()}

    return dt_classifier, pca, idx_to_class


if __name__ == "__main__":
    main()