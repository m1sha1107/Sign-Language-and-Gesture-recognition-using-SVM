import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the training script is tailored to image data and not sequences
# Removed any references to sequences and clarified that each image is treated independently

# Load processed data
def load_all_data():
    # Load all datasets
    features1 = np.load("data_features.npy")
    labels1 = np.load("data_labels.npy")
    features2 = np.load("new_data_features.npy")
    labels2 = np.load("new_data_labels.npy")

    # Combine datasets
    features = np.vstack((features1, features2))
    labels = np.concatenate((labels1, labels2))
    return features, labels

# Train the SVM model
def train_svm(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the SVM classifier
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return svm

# Save the trained model with a custom filename
def save_model(model, filename="svm_model1.joblib"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    features, labels = load_all_data()
    svm_model = train_svm(features, labels)
    save_model(svm_model)