# SVM-SignLanguage

## Overview
SVM-SignLanguage is a machine learning project designed to recognize hand signs and gestures from Indian Sign Language (ISL). The system uses Support Vector Machines (SVM) for classification and integrates OpenCV and MediaPipe for image and video processing. The project supports both static hand signs (e.g., alphabets and numbers) and dynamic gestures (e.g., "Thank you", "Yes", "No").

## Problem Statement

The primary goal of this project is to develop an efficient system for recognizing hand signs and gestures from Indian Sign Language (ISL). This system aims to bridge the communication gap between individuals who use sign language and those who do not understand it. By leveraging machine learning techniques, specifically Support Vector Machines (SVM), and integrating tools like OpenCV and MediaPipe, the project seeks to:

1. Recognize static hand signs (e.g., alphabets and numbers) with high accuracy.
2. Identify static gestures (e.g., "Thank you", "Yes", "No") in real-time.
3. Provide a real-time interface for practical use cases, such as education, accessibility, and communication.

## Features
- **Static Hand Sign Recognition**: Recognizes alphabets (A-Z) and numbers (0-9).
- **Static Gesture Recognition**: Recognizes common gestures like "Thank you", "Yes", "No", etc.
- **Real-Time Prediction**: Uses a webcam interface to predict hand signs and gestures in real-time.


## Project Structure
```
datasets/
    misha_dataset/               
    Misha_gesture_dataset/       
    nikhita_dataset/             
    spandanas_dataset/           
    spandana_gesture_dataset/    

SVM-SignLanguage/
    capture.py                   # Script for capturing images or videos
    prediction.py                # Real-time prediction script
    README.md                    # Project documentation
    requirements.txt             # Python dependencies
    train_svm.py                 # SVM training script
    data/                        # Processed data files
        data_features.npy
        data_labels.npy
        new_data_features.npy
        new_data_labels.npy
    models/                      # Trained SVM models
        svm_model.joblib
        svm_model1.joblib
    utils/
        preprocess.py            # Preprocessing script for datasets
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/SVM-SignLanguage.git
   cd SVM-SignLanguage
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocess the Datasets
Run the preprocessing script to extract features from the datasets:
```bash
python utils/preprocess.py
```

### 2. Train the SVM Model
Train the SVM model using the processed data:
```bash
python train_svm.py
```

### 3. Real-Time Prediction
Use the webcam interface to predict hand signs and gestures in real-time:
```bash
python prediction.py
```

### 4. Capture New Data (Optional)
Capture images or videos for new hand signs or gestures:
```bash
python capture.py
```

## Adding New Datasets
1. Place the new dataset in the `datasets/` directory.
2. Update the `utils/preprocess.py` script to include the new dataset.
3. Run the preprocessing script to extract features.
4. Retrain the SVM model using `train_svm.py`.

##Collaborators
1.Misha N Devegowda - m1sha1107
2.Nikhita K Nagavar
3.Spandana Sujay

## Acknowledgments
- **MediaPipe**: For hand landmark detection.
- **OpenCV**: For image and video processing.
- **scikit-learn**: For SVM implementation.
.