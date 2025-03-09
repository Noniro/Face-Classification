import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# === Step 1: Define variables ===
USE_PCA = True  # Set to False to disable PCA
USE_HOG = True  # Set to False to disable HOG
CHECK = True  # Set to True to display classification report
K_values = range(1, 6)  # Testing K from 1 to 5

# Function to extract HOG features
def extract_hog_features(X, pixels_per_cell=(8, 8), block_size=(2, 2)):
    hog_features = []
    for img in X:
        fd, _ = hog(img.reshape(64, 64), orientations=9, pixels_per_cell=pixels_per_cell,
                    cells_per_block=block_size, visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)

# === Step 2: Load the data ===
def load_images(root_dir, img_size=(64, 64)):
    images, labels = [], []
    class_names = sorted(os.listdir(root_dir))
    for label, folder in enumerate(class_names):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            images.append(img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels), class_names

# Load dataset
data_dir = "../Data-peopleFaces"
X, y, class_names = load_images(data_dir)

# === Step 3: Split data into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Step 4: Normalize the data ===
X_train = X_train / 255.0
X_test = X_test / 255.0

# Apply HOG if enabled
if USE_HOG:
    print("🔹 Using HOG...")
    X_train = extract_hog_features(X_train)
    X_test = extract_hog_features(X_test)

# PCA Setup
if USE_PCA:
    pca_dimensions = [10, 20, 50, 100, 200, 300, 500, 1000, 1500, X_train.shape[1]]

# Store results
results = {}

# === Step 5: Train KNN with different K values ===
for K in K_values:
    print(f"\n🔹 Training KNN with K={K}...")

    if USE_PCA:
        accuracy_results = []
        for n_components in pca_dimensions:
            print(f"  ↳ Using PCA with {n_components} components...")
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Train KNN model
            model = KNeighborsClassifier(n_neighbors=K)
            model.fit(X_train_pca, y_train)

            # Predict and evaluate
            y_pred = model.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_results.append(accuracy)
            print(f"    🎯 Accuracy with {n_components} components: {accuracy * 100:.2f}%")

        results[K] = accuracy_results
    else:
        # Train KNN without PCA
        model = KNeighborsClassifier(n_neighbors=K)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[K] = accuracy
        print(f"    🎯 Accuracy: {accuracy * 100:.2f}%")

        if CHECK:
            print(classification_report(y_test, y_pred))

# === Step 6: Plot results ===
plt.figure(figsize=(8, 5))
for K in K_values:
    if USE_PCA:
        plt.plot(pca_dimensions, results[K], marker='o', linestyle='--', label=f'K={K}')
    else:
        plt.scatter(K, results[K], label=f'K={K}', marker='o')

plt.xlabel("Number of PCA Components" if USE_PCA else "K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy with Different K values")
plt.legend()
plt.grid(True)
plt.show()
