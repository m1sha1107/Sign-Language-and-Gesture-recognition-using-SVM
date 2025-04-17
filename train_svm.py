import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load processed data
def load_all_data():
    # Load all datasets from the data directory
    features1 = np.load("data/data_features.npy") #For only ISL
    labels1 = np.load("data/data_labels.npy")
    features2 = np.load("data/new_data_features.npy") #For only Gestures
    labels2 = np.load("data/new_data_labels.npy")

    # Combine datasets
    features = np.vstack((features1, features2))
    labels = np.concatenate((labels1, labels2))
    return features, labels

# Training
def train_svm(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the SVM classifier
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)

    # Evaluate the model
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # Display class-wise accuracy
    print("\nClass-wise Accuracy:")
    report = classification_report(y_test, y_pred, target_names=np.unique(labels), zero_division=0)
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels)) #heatmap code
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return svm

def save_model(model, filename="svm_model1.joblib"): # byte stream
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    features, labels = load_all_data()
    svm_model = train_svm(features, labels)
    save_model(svm_model)