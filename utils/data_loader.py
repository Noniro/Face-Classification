import os
import cv2
import numpy as np

"""
From the kuggle fwl data set "The original images are 250 x 250 pixels,
but the default slice and resize arguments reduce them to 62 x 47 pixels.
"""

def load_data(dataset_path, img_size=(62, 47)):
    """
    Load images and labels from the dataset.

    Parameters:
    - dataset_path: str, path to the dataset directory.
    - img_size: tuple, size to which each image will be resized (width, height).

    Returns:
    - X: np.ndarray, array of images normalized to [0, 1].
    - y: np.ndarray, array of labels corresponding to the images.
    - label_dict: dict, mapping from label indices to person names.
    """
    images, labels, label_dict = [], [], {}
    person_classes = sorted(os.listdir(dataset_path))

    for label_idx, person_name in enumerate(person_classes):
        person_path = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_path):
            label_dict[label_idx] = person_name  # Store label mapping

            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                img = cv2.imread(image_path)

                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label_idx)

    X = np.array(images, dtype="float32") / 255.0  # Normalize images to [0, 1]
    y = np.array(labels)  # Array of labels , number for each person

    return X, y, label_dict
