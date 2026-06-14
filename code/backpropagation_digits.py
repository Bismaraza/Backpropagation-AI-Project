"""
Backpropagation from scratch for handwritten digit recognition.
Dataset: sklearn built-in digits dataset (8x8 grayscale images, 10 classes).
No deep learning framework is used. Only NumPy is used for the neural network math.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, seed=42):
        rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        # Xavier-style initialization for stable learning
        self.W1 = rng.normal(0, np.sqrt(2 / (input_size + hidden_size)), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / (hidden_size + output_size)), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def softmax(z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy(y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = X.shape[0]
        dz2 = (self.a2 - y_true) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def train(self, X_train, y_train, X_test, y_test, epochs=500):
        history = {"loss": [], "train_accuracy": [], "test_accuracy": []}
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X_train)
            loss = self.cross_entropy(y_train, y_pred)
            self.backward(X_train, y_train)

            train_pred = self.predict(X_train)
            test_pred = self.predict(X_test)
            train_acc = accuracy_score(y_train_labels, train_pred)
            test_acc = accuracy_score(y_test_labels, test_pred)

            history["loss"].append(loss)
            history["train_accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

            if epoch % 50 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        return history


def one_hot_encode(labels, num_classes):
    encoded = np.zeros((labels.size, num_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded


def plot_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"])
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_accuracy"], label="Training Accuracy")
    plt.plot(history["test_accuracy"], label="Testing Accuracy")
    plt.title("Training and Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_curve.png"), dpi=180)
    plt.close()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=180)
    plt.close()


def save_sample_predictions(images, y_true, y_pred):
    plt.figure(figsize=(10, 4))
    for idx in range(10):
        plt.subplot(2, 5, idx + 1)
        plt.imshow(images[idx], cmap="gray")
        plt.title(f"T:{y_true[idx]} P:{y_pred[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"), dpi=180)
    plt.close()


def main():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    original_images = digits.images

    X_train, X_test, y_train_labels, y_test_labels, img_train, img_test = train_test_split(
        X, y, original_images, test_size=0.2, random_state=42, stratify=y
    )

    y_train = one_hot_encode(y_train_labels, 10)
    y_test = one_hot_encode(y_test_labels, 10)

    model = NeuralNetwork(input_size=64, hidden_size=32, output_size=10, learning_rate=0.5, seed=42)
    history = model.train(X_train, y_train, X_test, y_test, epochs=500)

    predictions = model.predict(X_test)
    final_accuracy = accuracy_score(y_test_labels, predictions)
    cm = confusion_matrix(y_test_labels, predictions)
    report = classification_report(y_test_labels, predictions)

    plot_history(history)
    plot_confusion_matrix(cm)
    save_sample_predictions(img_test, y_test_labels, predictions)

    with open(os.path.join(RESULTS_DIR, "accuracy_report.txt"), "w", encoding="utf-8") as f:
        f.write("Backpropagation Neural Network Results\n")
        f.write("======================================\n")
        f.write(f"Dataset: sklearn digits handwritten digit dataset\n")
        f.write(f"Training samples: {X_train.shape[0]}\n")
        f.write(f"Testing samples: {X_test.shape[0]}\n")
        f.write(f"Input neurons: 64\n")
        f.write(f"Hidden neurons: 32\n")
        f.write(f"Output neurons: 10\n")
        f.write(f"Epochs: 500\n")
        f.write(f"Learning rate: 0.5\n")
        f.write(f"Final test accuracy: {final_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print("\nFinal Test Accuracy:", round(final_accuracy, 4))
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
