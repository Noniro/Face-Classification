from sklearn.model_selection import train_test_split
import numpy as np

# Splitting the data into training and testing sets.
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    num_classes = len(set(y))
    y_train_cat = np.zeros((y_train.size, num_classes))
    y_train_cat[np.arange(y_train.size), y_train] = 1

    y_test_cat = np.zeros((y_test.size, num_classes))
    y_test_cat[np.arange(y_test.size), y_test] = 1

    return X_train, X_test, y_train_cat, y_test_cat
# y_train_cat and y_test_cat - [num_samples, num_classes] one-hot encoded labels (each row is the class label for a sample).