import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import pyttsx3
from collections import Counter
# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("models/gesture_model.pkl")

labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

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
prev_time = 0
prediction_buffer = []
BUFFER_SIZE = 10
# -----------------------------
# Word Builder
# -----------------------------
current_word = ""
current_sentence = ""
engine = pyttsx3.init()

engine.setProperty("rate", 150)      # Speed
engine.setProperty("volume", 1.0)    # Volume
space_pressed = False
last_added_letter = ""

while True:

    success, frame = cap.read()
    current_time = time.time()

    fps = int(1 / (current_time - prev_time)) if current_time != prev_time else 0

    prev_time = current_time


    if not success:
        break

    frame = cv2.flip(frame, 1)
    # Create left dashboard
    dashboard = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)

    # Combine dashboard + webcam
    frame = np.hstack((dashboard, frame))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        h, w, _ = frame.shape

        x_list = []
        y_list = []

        for lm in hand.landmark:
            x_list.append(int(lm.x * w))
            y_list.append(int(lm.y * h))

        x_min = max(min(x_list) - 20, 0)
        x_max = min(max(x_list) + 20, w)

        y_min = max(min(y_list) - 20, 0)
        y_max = min(max(y_list) + 20, h)

        cv2.rectangle(
            frame,
            (x_min + 300, y_min),
            (x_max + 300, y_max),
            (0, 255, 0),
            2
        )

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([
                lm.x,
                lm.y,
                lm.z
            ])


        sample = np.array(landmarks).reshape(1, -1)

        prediction = model.predict(sample)[0]

        confidence = np.max(model.predict_proba(sample)) * 100

        prediction_buffer.append(prediction)

        if len(prediction_buffer) > BUFFER_SIZE:
            prediction_buffer.pop(0)

        stable_prediction = Counter(prediction_buffer).most_common(1)[0][0]

        if confidence >= 85:
            letter = labels[stable_prediction]
            color = (0, 255, 0)
        else:
            letter = "Unknown"
            color = (0, 0, 255)


        cv2.putText(frame, "GestureSense AI", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"Letter : {letter}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"Confidence : {confidence:.2f}%", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Status : Hand Detected", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS : {fps}", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "AI Model : MLP", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Dataset : 14614", (20, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Classes : 26", (20, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "S = Speak",
            (20, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )
        cv2.putText(
            frame,
            "Enter = Save Word",
            (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Space = Add Letter",
            (20, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Word : {current_word}",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Sentence : {current_sentence}",
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(frame,
                    "SPACE = Add Letter",
                    (20, 500),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

        cv2.putText(frame,
                    "C = Clear",
                    (20, 530),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

    else:

        cv2.putText(
            frame,
            "GestureSense AI",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Status : No Hand Detected",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
    cv2.putText(
        frame,
        f"FPS : {fps}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    cv2.imshow("GestureSense AI", frame)

    key = cv2.waitKey(1) & 0xFF

    # Space = Add current letter
    if key == ord(" "):

        if not space_pressed:

            if letter != "Unknown":
                current_word += letter

            space_pressed = True



    # Enter = Add word to sentence
    elif key == 13:  # Enter key

        if current_word != "":

            if current_sentence == "":
                current_sentence = current_word
            else:
                current_sentence += " " + current_word

            current_word = ""

    # C = Clear everything
    elif key == ord("c"):

        current_word = ""
        current_sentence = ""
    # S = Speak sentence
    elif key == ord("s"):

        if current_sentence != "":
            engine.say(current_sentence)
            engine.runAndWait()
    # Backspace = Delete last letter
    elif key == 8:

        if len(current_word) > 0:
            current_word = current_word[:-1]

    # Q = Quit
    elif key == ord("q"):
        break
    else:
        space_pressed = False
cap.release()
cv2.destroyAllWindows()