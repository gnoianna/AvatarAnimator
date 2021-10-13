import cv2
import dlib
import imutils
import numpy as np
from PIL import Image
from flask import Flask, render_template, Response, request, send_file
from imutils import face_utils

global out, detector, predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)


def gen_frames_with_landmarks():
    global out, detector, predictor

    while True:
        success, frame = camera.read()
        frame = imutils.resize(frame)
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                for (x, y) in shape:
                    frame = cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass

        else:
            pass


def gen_avatar_stream():
    img = cv2.imread("./static/awatar.jpg")
    new_img = gen_landmarks(img)

    # try:
    #     return send_file(
    #         new_img,
    #         as_attachment=True,
    #         attachment_filename='./static/awatar.jpg')
    # except FileNotFoundError as e:
    #     print(e)

    try:
        ret, buffer = cv2.imencode('.jpg', new_img)
        new_img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + new_img + b'\r\n')
    except Exception as e:
        pass

    else:
        pass


def gen_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            image = cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/camera_image')
def camera_image():
    return Response(gen_frames_with_landmarks(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/avatar_image')
def avatar_image():
    return Response(gen_avatar_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
