import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# === Step 1: Set the variable to control PCA ===
USE_PCA = True  # Change to False if you don't want to use PCA
USE_HOG = True  # Change to True if you want to use HOG
CHECK = False


# Function to use HOG
def extract_hog_features(X, cell_size=(8, 8), block_size=(2, 2), pixels_per_cell=(8, 8)):
    hog_features = []
    for img in X:
        fd, _ = hog(img.reshape(64, 64), orientations=9, pixels_per_cell=pixels_per_cell, cells_per_block=block_size,
                    visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)


# === Step 2: Load the data ===
def load_images(root_dir, img_size=(64, 64)):
    images, labels = [], []
    class_names = sorted(os.listdir(root_dir))  # List of classes

    for label, folder in enumerate(class_names):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # Ignore irrelevant files

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale image
            if img is None:
                print(f"⚠️ Error loading {img_path}")
                continue

            img = cv2.resize(img, img_size)  # Resizing
            images.append(img.flatten())  # Flatten the image into a vector
            labels.append(label)  # Get the class number (0, 1, 2...)

    return np.array(images), np.array(labels), class_names


# Define the path for images
data_dir = "../Data-peopleFaces"  # Update the path accordingly
X, y, class_names = load_images(data_dir)

# === Step 3: Split the data into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Step 4: Normalize the data (values between 0 and 1) ===
X_train = X_train / 255.0
X_test = X_test / 255.0

if USE_HOG:
    print("🔹 Using HOG...")
    X_train = extract_hog_features(X_train)
    X_test = extract_hog_features(X_test)

# === Step 5: Test with different PCA dimensions (if enabled) ===
if USE_PCA:
    if USE_HOG:
        pca_dimensions = [10, 20, 50, 100, 200, 300, 500, 1000, 1500, 1764]  # Can change accordingly
    else:
        pca_dimensions = [10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 2500, 2760]  # Can change accordingly
    accuracy_results = []

    for n_components in pca_dimensions:
        print(f"🔹 Using PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # === Step 6: Train AdaBoost model ===
        # works with decision trees as base learners
        model = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
        model.fit(X_train_pca, y_train)

        # === Step 7: Prediction on the test set ===
        y_pred = model.predict(X_test_pca)

        # === Step 8: Model evaluation ===
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results.append(accuracy)

        print(f"🎯 Accuracy with {n_components} components: {accuracy * 100:.2f}%")

    # Display the graph of accuracy vs PCA components
    plt.figure(figsize=(8, 5))
    plt.plot(pca_dimensions, accuracy_results, marker='o', linestyle='--', color='b')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Accuracy")
    technics = "HOG" if USE_HOG else ""
    plt.title("Accuracy vs PCA Components and " + technics + " with AdaBoost")
    plt.grid(True)
    plt.show()

else:
    # If PCA is not enabled, use raw data
    print("🔹 Training AdaBoost model without PCA...")
    model = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
    model.fit(X_train, y_train)

    # === Step 7: Prediction on the test set ===
    y_pred = model.predict(X_test)

    # === Step 8: Model evaluation ===
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model accuracy: {accuracy * 100:.2f}%")

if CHECK:
    # Display detailed performance report
    print(classification_report(y_test, y_pred))
