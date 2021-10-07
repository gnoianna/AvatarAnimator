import cv2
import dlib
import imutils
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
        frame = imutils.resize(frame, height=500)
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
    img = './static/awatar.jpg'

    try:
        return send_file(
            img,
            as_attachment=True,
            attachment_filename='./static/pies.jpg',
            mimetype='image/jpeg')
    except FileNotFoundError as e:
        print(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/camera_image')
def camera_image():
    return Response(gen_frames_with_landmarks(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/avatar_image')
def avatar_image():
    return gen_avatar_stream()


if __name__ == '__main__':
    app.run()
