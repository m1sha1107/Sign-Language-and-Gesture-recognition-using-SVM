import cv2
import os

# Define the number of images per class
NUM_IMAGES = 2000 
SIGN_CLASSES = ["Like"]
DATASET_PATH = "datasets/Misha_gesture_dataset"

# Create dataset folder if it doesn't exist
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Start capturing for each sign
cap = cv2.VideoCapture(0)

for sign in SIGN_CLASSES:
    sign_path = os.path.join(DATASET_PATH, sign)
    if not os.path.exists(sign_path):
        os.makedirs(sign_path)

    print(f" Collecting images for: {sign}")
    count = 0

    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw instructions
        cv2.putText(frame, f"Collecting: {sign} ({count}/{NUM_IMAGES})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        # Save frame to dataset folder
        img_path = os.path.join(sign_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

print("Dataset Collection Complete")
cap.release()
cv2.destroyAllWindows()