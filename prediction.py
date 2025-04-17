import cv2
import mediapipe as mp
import joblib
import numpy as np

#loading the model :)
model = joblib.load("svm_model1.joblib")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) #IMAGES ARE NOT STATIC HERE!!

#landmark extraction function
def extract_landmarks_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    return None

#prediction
def real_time_prediction():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract landmarks
        landmarks = extract_landmarks_from_frame(frame)

        if landmarks is not None: #prediction
            prediction = model.predict([landmarks])[0]

            # Displaying the prediction
            cv2.putText(frame, f"Prediction: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        
        cv2.imshow("Sign Language Recognition", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_prediction()