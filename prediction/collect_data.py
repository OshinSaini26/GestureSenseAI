import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# Dataset Folder
# -----------------------------
DATASET_PATH = "../dataset/csv"
os.makedirs(DATASET_PATH, exist_ok=True)

# -----------------------------
# Letters
# -----------------------------
LETTERS = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

current_letter_index = 0
current_letter = LETTERS[current_letter_index]

sample_count = 0
TARGET = 500

# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # Draw Hand
    # -----------------------------
    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Save while holding current letter key
        if key == ord(current_letter.lower()):

            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            csv_file = os.path.join(
                DATASET_PATH,
                f"{current_letter}.csv"
            )

            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)

            sample_count += 1

    # -----------------------------
    # Next Letter
    # -----------------------------
    if key == ord(']'):

        if current_letter_index < len(LETTERS) - 1:

            current_letter_index += 1
            current_letter = LETTERS[current_letter_index]
            sample_count = 0

            print(f"\nNow Collecting {current_letter}")

    # -----------------------------
    # Reset Count
    # -----------------------------
    if key == ord('/'):

        sample_count = 0

        csv_file = os.path.join(
            DATASET_PATH,
            f"{current_letter}.csv"
        )

        open(csv_file, "w").close()

        print(f"{current_letter} reset.")

    # -----------------------------
    # Display
    # -----------------------------
    cv2.putText(
        frame,
        f"Letter : {current_letter}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Samples : {sample_count}/{TARGET}",
        (10,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.putText(
        frame,
        f"Hold '{current_letter}' key to save",
        (10,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        frame,
        "] = Next Letter",
        (10,135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        frame,
        "/ = Reset",
        (10,170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,0,255),
        2
    )

    cv2.imshow("GestureSense AI - Data Collection", frame)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()