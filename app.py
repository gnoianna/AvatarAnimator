import cv2
import dlib
import imutils
import numpy as np
from flask import Flask, render_template, Response, request
from imutils import face_utils
from skimage.transform import PiecewiseAffineTransform, warp

global detector, predictor, avatar_temp_image, avatar_original_image, animate, file_name, camera


app = Flask(__name__, template_folder='./templates')


def animate_avatar(avatar_img, avatar_points, src_img, src_points):
    global avatar_temp_image

    avatar_new_face = np.zeros(src_img.shape, np.uint8)
    avatar_img_copy = np.copy(avatar_img)

    for indices in gen_triangles_indices(src_points):
        avatar_triangle_points = gen_xy_from_indices(avatar_points, indices)
        src_triangle_points = gen_xy_from_indices(src_points, indices)

        avatar_triangle, avatar_img_cropped, _ = gen_cropped_triangle(avatar_img_copy, avatar_triangle_points)
        src_triangle, src_img_cropped, (x, y, w, h) = gen_cropped_triangle(src_img, src_triangle_points)

        transform_matrix = cv2.getAffineTransform(np.float32(avatar_triangle), np.float32(src_triangle))

        avatar_t_warped = cv2.warpAffine(avatar_img_cropped, transform_matrix,
                                         (src_img_cropped.shape[1], src_img_cropped.shape[0]), None,
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros(src_img_cropped.shape, dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, np.int32(src_triangle), (1.0, 1.0, 1.0), 16, 0)

        avatar_t_warped *= mask

        new_face_gray = cv2.cvtColor(avatar_new_face[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        _, mask_triangles = cv2.threshold(new_face_gray, 0, 255, cv2.THRESH_BINARY_INV)
        avatar_t_warped = cv2.bitwise_and(avatar_t_warped, avatar_t_warped, mask=mask_triangles)

        avatar_new_face[y: y + h, x: x + w] += avatar_t_warped

    transform = PiecewiseAffineTransform()
    transform.estimate(avatar_points[0:35], src_points[0:35])
    face = warp(avatar_new_face, transform, output_shape=avatar_img.shape, order=1, mode='wrap')
    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(face_gray, 1, 255, cv2.THRESH_BINARY_INV)
    new_avatar_img = cv2.bitwise_and(avatar_img, avatar_img, mask=mask)

    new_avatar_img += face

    avatar_temp_image = new_avatar_img


def gen_frames_with_landmarks():
    global detector, predictor, avatar_temp_image, animate, avatar_original_image

    while True:
        success, frame = camera.read()
        frame = imutils.resize(frame)
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            rects = [item for item in rects]
            if rects:
                for rect in rects:
                    camera_shape = predictor(gray, rect)
                    camera_shape = face_utils.shape_to_np(camera_shape)
                    if animate:
                        animate = not animate
                        avatar_original_image, avatar_shape = gen_landmarks(avatar_original_image, 0)
                        temp_avatar = np.copy(avatar_original_image)
                        animate_avatar(temp_avatar, avatar_shape, frame, camera_shape)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(e)

        else:
            pass


def gen_avatar_stream():
    global avatar_temp_image

    while True:
        try:
            _, encoded_avatar = cv2.imencode(".jpg", avatar_temp_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_avatar) + b'\r\n')
        except Exception as e:
            print(e)

        else:
            pass


def gen_landmarks(img, if_with_landmarks):
    global detector, predictor

    without_landmarks_img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        img_shape = predictor(gray, rect)
        img_shape = face_utils.shape_to_np(img_shape)

        for i, (x, y) in enumerate(img_shape):
            img = cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    if if_with_landmarks:
        return img, img_shape
    else:
        return without_landmarks_img, img_shape


def gen_triangles_indices(points):
    b_rect = (*points.min(axis=0), *points.max(axis=0))

    subdiv_object = cv2.Subdiv2D(b_rect)
    points_list = list(points)
    subdiv_object.insert(points_list)
    triangle_list = subdiv_object.getTriangleList()

    for x1, y1, x2, y2, x3, y3 in triangle_list:
        yield [(points == point).all(axis=1).nonzero()[0][0] for point in [(x1, y1), (x2, y2), (x3, y3)]]


def gen_cropped_triangle(img, triangle):
    b_rect = cv2.boundingRect(triangle)
    img_cropped = img[b_rect[1]:b_rect[1] + b_rect[3], b_rect[0]:b_rect[0] + b_rect[2]]

    triangle_cropped = [(indices[0] - b_rect[0], indices[1] - b_rect[1]) for indices in triangle]

    return triangle_cropped, img_cropped, b_rect


def gen_xy_from_indices(points, indices):
    triangles = [points[single_index] for single_index in indices]

    return np.array(triangles)


def update_avatar():
    global avatar_temp_image, avatar_original_image, file_name

    avatar_original_image = cv2.imread(file_name)


def init():
    global predictor, detector, animate, file_name, camera, avatar_temp_image, avatar_original_image

    animate = 0
    camera = cv2.VideoCapture(0)
    file_name = "./static/sim_1.jpg"
    avatar_temp_image = cv2.imread(file_name)
    avatar_original_image = cv2.imread(file_name)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')


@app.route('/')
def index():
    init()
    return render_template('index.html')


@app.route('/camera_image')
def camera_image():
    print("camera_image")
    return Response(gen_frames_with_landmarks(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/avatar_image')
def avatar_image():
    global avatar_temp_image, animate
    print("avatar_image")

    return Response(gen_avatar_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global animate, file_name, avatar_temp_image
    if request.method == 'POST':

        if request.form.get("avatar"):
            file_name = "." + request.form.get("avatar")

        animate = request.form.get("animate")
        update_avatar()

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
