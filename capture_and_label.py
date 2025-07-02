import cv2
import os

# Define emotion labels and corresponding keys
EMOTIONS = {
    'h': 'happy',
    'n': 'neutral',
    's': 'sad'
}

# Create directories for each emotion if not exist
dataset_dir = 'dataset'
for label in EMOTIONS.values():
    os.makedirs(os.path.join(dataset_dir, label), exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Counters for saved images
counters = {label: len(os.listdir(os.path.join(dataset_dir, label))) for label in EMOTIONS.values()}

print("[INFO] Camera capture started.")
print("Press 'h' for Happy, 'n' for Neutral, 's' for Sad. Press 'q' to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capture & Label (h/n/s/q)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in map(ord, EMOTIONS.keys()) and len(faces) > 0:
        emotion = EMOTIONS[chr(key)]
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))
        count = counters[emotion]
        save_path = os.path.join(dataset_dir, emotion, f"{count}.jpg")
        cv2.imwrite(save_path, face_img)
        counters[emotion] += 1
        print(f"[SAVED] {emotion} #{count}")

cap.release()
cv2.destroyAllWindows()
