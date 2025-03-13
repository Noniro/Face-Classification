import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# === Step 1: Set Variables ===
USE_PCA = False  # Enable/disable PCA
USE_HOG = True  # Enable/disable HOG features
CHECK = False  # Print classification report


# === Function to Extract HOG Features ===
def extract_hog_features(X, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features = []
    for img in X:
        fd, _ = hog(img.reshape(64, 64), orientations=9, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=True)
        hog_features.append(fd)
    return np.array(hog_features)


# === Function to Load Images ===
def load_images(root_dir, img_size=(64, 64)):
    images, labels = [], []
    class_names = sorted(os.listdir(root_dir))  # Class names from folders

    for label, folder in enumerate(class_names):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # Ignore files

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Error loading {img_path}")
                continue

            img = cv2.resize(img, img_size)
            images.append(img.flatten())  # Flatten image
            labels.append(label)  # Assign class label

    return np.array(images), np.array(labels), class_names


# === Load Data ===
data_dir = "../Data-peopleFaces"  # Adjust path
X, y, class_names = load_images(data_dir)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Normalize Data ===
X_train = X_train / 255.0
X_test = X_test / 255.0

if USE_HOG:
    print("🔹 Using HOG features...")
    X_train = extract_hog_features(X_train)
    X_test = extract_hog_features(X_test)

pca_dimensions = []

# === PCA Setup ===
if USE_PCA:
    if USE_HOG:
        pca_dimensions = [10, 20, 50, 100, 200, 300, 500, 1000, 1500, 1764]  # Can change as needed
    else:
        pca_dimensions = [10, 20, 50, 100, 200, 300, 500, 1000, 1500, 2000, 2500, 2760]  # Can change as needed


# === AdaBoost Classifiers ===
base_classifiers = {
    "Decision Tree (Default)": DecisionTreeClassifier(max_depth=1),
    "SVM": SVC(probability=True, kernel='linear'),
    "Logistic Regression": LogisticRegression(max_iter=200),
}

# === Train and Evaluate Each Classifier ===
# === Train and Evaluate Each Classifier ===
results = []
if USE_PCA:
    # Loop over PCA dimensions if PCA is being used
    for n_components in pca_dimensions:
        if n_components:
            print(f"\n🔹 Applying PCA with {n_components} components...")
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
        else:
            X_train_pca, X_test_pca = X_train, X_test  # No PCA, use original data

        # Train and evaluate each classifier
        for clf_name, base_estimator in base_classifiers.items():
            print(f"\n🚀 Training AdaBoost with {clf_name} (Algorithm: SAMME)...")
            model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, algorithm="SAMME", random_state=42)
            model.fit(X_train_pca, y_train)

            # Predictions
            y_pred = model.predict(X_test_pca)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results.append((clf_name, n_components, accuracy))

            print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

            # Classification report (optional)
            if CHECK:
                print(classification_report(y_test, y_pred))
else:
    # If no PCA is used, train directly on the original data
    X_train_pca, X_test_pca = X_train, X_test  # No PCA, use original data
    for clf_name, base_estimator in base_classifiers.items():
        print(f"\n🚀 Training AdaBoost with {clf_name} (Algorithm: SAMME)...")
        model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, algorithm="SAMME", random_state=42)
        model.fit(X_train_pca, y_train)

        # Predictions
        y_pred = model.predict(X_test_pca)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results.append((clf_name, None, accuracy))  # No PCA, so None is used

        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

        # Classification report (optional)
        if CHECK:
            print(classification_report(y_test, y_pred))

# === Display Final Results ===
print("\n📊 Final Results:")
for clf_name, n_components, accuracy in results:
    pca_str = f"{n_components} components" if n_components else "No PCA"
    print(f"{clf_name} | {pca_str} → Accuracy: {accuracy * 100:.2f}%")

# === Plot Accuracy Results ===
plt.figure(figsize=(10, 6))
for clf_name in base_classifiers:
    algo_results = [acc for c_name, pca, acc in results if c_name == clf_name]
    plt.plot(pca_dimensions if USE_PCA else [None], algo_results, marker='o', linestyle='--', label=clf_name)

plt.xlabel("Number of PCA Components" if USE_PCA else "No PCA")
plt.ylabel("Accuracy")
plt.title("Accuracy vs PCA Components for Different AdaBoost Classifiers")
plt.legend()
plt.grid(True)
plt.show()
