import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to extract landmarks from an image
def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

# Function to process a dataset directory
def process_dataset(dataset_path):
    data = []
    labels = []

    for label in tqdm(os.listdir(dataset_path), desc="Processing labels"):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        for image_name in tqdm(os.listdir(label_path), desc=f"Processing images for label {label}", leave=False):
            image_path = os.path.join(label_path, image_name)
            landmarks = extract_landmarks(image_path)
            if landmarks is not None:
                data.append(landmarks)
                labels.append(label)

    return np.array(data), np.array(labels)

# Main function to process specific datasets
def prepare_specific_datasets():
    # Update the base path to an absolute path
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
    datasets = ["Misha_gesture_dataset", "spandana_gesture_dataset"]

    all_data = []
    all_labels = []

    for dataset in tqdm(datasets, desc="Processing specific datasets"):
        dataset_path = os.path.join(base_path, dataset)
        data, labels = process_dataset(dataset_path)
        all_data.append(data)
        all_labels.append(labels)

    # Checks for empty datasets before concatenation
    all_data = [data for data in all_data if data.size > 0]
    all_labels = [labels for labels in all_labels if labels.size > 0]

    if not all_data or not all_labels:
        print("No valid data found in the specified datasets.")
        return

    all_data = np.vstack(all_data)
    all_labels = np.concatenate(all_labels)

    # Save processed data
    np.save(os.path.join(base_path, "..", "SVM-SignLanguage", "new_data_features.npy"), all_data)
    np.save(os.path.join(base_path, "..", "SVM-SignLanguage", "new_data_labels.npy"), all_labels)

if __name__ == "__main__":
    prepare_specific_datasets()