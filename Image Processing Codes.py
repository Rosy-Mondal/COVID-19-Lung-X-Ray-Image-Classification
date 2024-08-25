import os
import numpy as np
from collections import Counter
from PIL import Image, ImageFilter, ImageEnhance
import random

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    label_map = {}
    current_label = 0
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            if subfolder not in label_map:
                label_map[subfolder] = current_label
                current_label += 1
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((64, 64))  # Resize image

                # Apply some filtering techniques (e.g., Gaussian blur)
                img = img.filter(ImageFilter.GaussianBlur(radius=2))

                # Apply some enhancement techniques (e.g., contrast enhancement)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)

                # Apply some segmentation techniques (e.g., thresholding)
                threshold = 100
                img = img.point(lambda p: p > threshold and 255)

                img = np.array(img).flatten()  # Flatten image
                images.append(img)
                labels.append(label_map[subfolder])
    return np.array(images), np.array(labels), label_map

# Function to split dataset
def train_test_split(X, y, test_size=0.2):
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_indices = indices[:split]
    test_indices = indices[split:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Function to calculate confusion matrix and metrics
def evaluate(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)

    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    TN = np.sum(confusion_matrix) - (TP + FP + FN)

    accuracy = np.sum(TP) / np.sum(confusion_matrix)
    precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP, dtype=float), where=(TP + FP) != 0)
    recall = np.divide(TP, (TP + FN), out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    f1_score = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(precision, dtype=float), where=(precision + recall) != 0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    return confusion_matrix, accuracy, precision, recall, f1_score

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))
        self.priors = np.zeros(len(self.classes))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        posteriors = []
        for x in X:
            posteriors.append(self._predict(x))
        return np.array(posteriors)

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Decision Tree Classifier
class DecisionTreeClassifier:
    class Node:
        def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
            self.gini = gini
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = 0
            self.threshold = 0
            self.left = None
            self.right = None

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(
            gini=self._gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

# K-Nearest Neighbors Classifier
class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load dataset
X, y, label_map = load_images_from_folder("C:/Users/rosym/Downloads/python/image_dataset")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train and evaluate Naive Bayes
nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("Naive Bayes Classifier:")
confusion_matrix_nb, accuracy_nb, precision_nb, recall_nb, f1_score_nb = evaluate(y_test, y_pred_nb, len(label_map))

# Train and evaluate Decision Tree
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Classifier:")
confusion_matrix_dt, accuracy_dt, precision_dt, recall_dt, f1_score_dt = evaluate(y_test, y_pred_dt, len(label_map))

# Train and evaluate K-Nearest Neighbors
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\nK-Nearest Neighbors Classifier:")
confusion_matrix_knn, accuracy_knn, precision_knn, recall_knn, f1_score_knn = evaluate(y_test, y_pred_knn, len(label_map))
