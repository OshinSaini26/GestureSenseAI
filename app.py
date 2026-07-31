import cv2
import os
import signal
import threading

from utils.predictor import process_frame, get_prediction_data
from flask import Flask, render_template, Response, jsonify
app = Flask(__name__)

camera = cv2.VideoCapture(0)

@app.route("/exit", methods=["POST"])
def exit_app():

    def shutdown():
        camera.release()          # Release webcam
        os.kill(os.getpid(), signal.SIGINT)

    threading.Timer(1, shutdown).start()

    return "Application Closed"

@app.route("/prediction")
def prediction():
    return jsonify(get_prediction_data())

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame +
               b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
@app.route("/add_letter", methods=["POST"])
def add_letter_route():
    from utils.predictor import add_current_letter
    add_current_letter()
    return "OK"
@app.route("/add_word", methods=["POST"])
def add_word_route():

    from utils.predictor import add_current_word

    add_current_word()

    return "OK"

@app.route("/clear", methods=["POST"])
def clear_route():

    from utils.predictor import clear_text

    clear_text()

    return "OK"

@app.route("/speak", methods=["POST"])
def speak_route():

    from utils.predictor import speak_sentence

    speak_sentence()

    return "OK"
@app.route("/backspace", methods=["POST"])
def backspace_route():

    from utils.predictor import backspace

    backspace()

    return "OK"
if __name__ == "__main__":
    app.run(debug=True)