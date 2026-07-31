import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import pyttsx3
from collections import Counter

# Load model
model = joblib.load("models/gesture_model.pkl")

labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# MediaPipe
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Prediction buffer
prediction_buffer = []
BUFFER_SIZE = 10

# Word builder
current_word = ""
current_sentence = ""

engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

space_pressed = False
latest_letter = "None"
latest_confidence = 0.0
def process_frame(frame):

    global prediction_buffer
    global current_word
    global current_sentence
    global space_pressed

    frame = cv2.flip(frame, 1)

    #dashboard = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)

    #frame = np.hstack((dashboard, frame))

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
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )
        # -----------------------------
        # Extract landmarks
        # -----------------------------
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

        global latest_letter
        global latest_confidence

        latest_letter = letter
        latest_confidence = confidence

        cv2.putText(
            frame,
            "GestureSense AI",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Letter : {letter}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence:.2f}%",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Status : Hand Detected",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    else:

        cv2.putText(
            frame,
            "GestureSense AI",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Status : No Hand Detected",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return frame
def get_prediction_data():

    return {

        "letter": latest_letter,
        "confidence": round(latest_confidence,2),
        "word": current_word,
        "sentence": current_sentence

    }
def add_current_letter():
    global current_word
    global latest_letter

    if latest_letter != "Unknown" and latest_letter != "None":
        current_word += latest_letter

def add_current_word():
    global current_word
    global current_sentence

    if current_word != "":

        if current_sentence == "":
            current_sentence = current_word
        else:
            current_sentence += " " + current_word

        current_word = ""


def clear_text():

    global current_word
    global current_sentence

    current_word = ""
    current_sentence = ""

def speak_sentence():

    global current_sentence

    if current_sentence != "":
        engine.say(current_sentence)
        engine.runAndWait()
def backspace():

    global current_word

    if len(current_word)>0:
        current_word=current_word[:-1]