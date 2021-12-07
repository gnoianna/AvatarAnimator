import cv2
import dlib
import imutils
import numpy as np
from flask import Flask, render_template, Response, request
from imutils import face_utils
from skimage.transform import PiecewiseAffineTransform, warp

global detector, predictor, avatar_temp_img, avatar_original_img, animate, file_name, camera

app = Flask(__name__, template_folder='./templates')


def init():
    global predictor, detector, animate, file_name, camera, avatar_temp_img, avatar_original_img

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

    file_name = "./static/1.png"
    animate, camera = 0, cv2.VideoCapture(0)
    avatar_temp_img, avatar_original_img = cv2.imread(file_name), cv2.imread(file_name)


def choose_specific_points(points):
    new_points = np.array(points[0:30])
    np.append(new_points, points[36])
    np.append(new_points, points[39])
    np.append(new_points, points[42])
    np.append(new_points, points[45])
    print(new_points)
    return new_points


def generate_new_face(avatar_img, avatar_points, src_points, avatar_new_face):
    transform = PiecewiseAffineTransform()
    transform.estimate(choose_specific_points(avatar_points), choose_specific_points(src_points))

    face = warp(avatar_new_face, transform, output_shape=avatar_img.shape, order=0, mode='wrap')
    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(face_gray, 0, 255, cv2.THRESH_BINARY_INV)
    new_avatar_img = cv2.bitwise_and(avatar_img, avatar_img, mask=mask)

    new_avatar_img += face

    return new_avatar_img


def add_triangle_to_new_face(new_face, warped_triangle, b_rect):
    (x, y, w, h) = b_rect
    new_face_gray = cv2.cvtColor(new_face[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(new_face_gray, 0, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask)

    new_face[y: y + h, x: x + w] += warped_triangle


def transform_triangle(avatar_triangle, avatar_img_cropped, src_triangle, src_img_cropped):

    transform_matrix = cv2.getAffineTransform(np.float32(avatar_triangle), np.float32(src_triangle))

    warped_triangle = cv2.warpAffine(avatar_img_cropped, transform_matrix,
                                     (src_img_cropped.shape[1], src_img_cropped.shape[0]), None,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros(src_img_cropped.shape, dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, np.int32(src_triangle), (1.0, 1.0, 1.0), 16, 0)
    warped_triangle *= mask

    return warped_triangle


def animate_avatar(avatar_img, avatar_points, src_img, src_points):
    new_face = np.zeros(src_img.shape, np.uint8)

    for src_triangle_points, avatar_triangle_points in delaunay_triangulation(src_points, avatar_points):
        avatar_img_cropped, avatar_triangle, _ = crop_single_triangle(avatar_img, avatar_triangle_points)
        src_img_cropped, src_triangle, b_rect = crop_single_triangle(src_img, src_triangle_points)

        warped_triangle = transform_triangle(avatar_triangle, avatar_img_cropped, src_triangle, src_img_cropped)
        add_triangle_to_new_face(new_face, warped_triangle, b_rect)

    return generate_new_face(avatar_img, avatar_points, src_points, new_face)


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


def delaunay_triangulation(src_points, avatar_points):
    points_list = list(src_points)
    delaunay_subdivision = cv2.Subdiv2D((*src_points.min(axis=0), *src_points.max(axis=0)))
    delaunay_subdivision.insert(points_list)

    for x_1, y_1, x_2, y_2, x_3, y_3 in delaunay_subdivision.getTriangleList():
        index_tuple = [(src_points == single_point).all(axis=1).nonzero()[0][0]
                       for single_point in [[x_1, y_1], [x_2, y_2], [x_3, y_3]]]

        src_triangles = gen_xy_for_indices(src_points, index_tuple)
        avatar_triangles = gen_xy_for_indices(avatar_points, index_tuple)
        yield [np.array(src_triangles), np.array(avatar_triangles)]


def gen_xy_for_indices(points, indices):
    return [points[single_index] for single_index in indices]


def crop_single_triangle(img, triangle):
    b_rect = cv2.boundingRect(triangle)
    triangle_cropped = [(indices[0] - b_rect[0], indices[1] - b_rect[1]) for indices in triangle]
    img_cropped = img[b_rect[1]:b_rect[1] + b_rect[3], b_rect[0]:b_rect[0] + b_rect[2]]

    return img_cropped, triangle_cropped, b_rect


def draw(img, triangle_list):
    t = triangle_list
    cv2.line(img, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(img, (int(t[2]), int(t[3])), (int(t[4]), int(t[5])), (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(img, (int(t[4]), int(t[5])), (int(t[0]), int(t[1])), (255, 255, 255), 1, cv2.LINE_AA, 0)


def draw_triangles(src_img, src_points, dst_img, dst_points):
    for src_triangle_points, dst_triangle_points in app.delaunay_triangulation(src_points, dst_points):
        t_s = [val for sublist in src_triangle_points for val in sublist]
        t_d = [val for sublist in dst_triangle_points for val in sublist]
        draw(src_img, t_s)
        draw(dst_img, t_d)


def generate_camera_stream():
    global detector, predictor, avatar_temp_img, animate, avatar_original_img

    while True:
        success, frame = camera.read()
        frame = imutils.resize(frame)
        if animate:
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                if rects:
                    for rect in rects:
                        camera_shape = predictor(gray, rect)
                        camera_shape = face_utils.shape_to_np(camera_shape)

                        animate = not animate
                        avatar_original_img, avatar_shape = gen_landmarks(avatar_original_img, 0)
                        temp_avatar = np.copy(avatar_original_img)
                        avatar_temp_img = animate_avatar(temp_avatar, avatar_shape, frame, camera_shape)

        try:
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(e)

        else:
            pass


def generate_avatar_stream():
    global avatar_temp_img

    while True:
        try:
            _, encoded_avatar = cv2.imencode(".jpg", avatar_temp_img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_avatar) + b'\r\n')
        except Exception as e:
            print(e)

        else:
            pass


@app.route('/')
def index():
    init()
    return render_template('index.html')


@app.route('/camera_image')
def camera_image():
    print("camera_image")
    return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/avatar_image')
def avatar_image():
    global avatar_temp_img, animate
    print("avatar_image")

    return Response(generate_avatar_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/change_avatar', methods=['POST', 'GET'])
def change_avatar():
    global animate, file_name, avatar_temp_img, avatar_original_img
    if request.method == 'POST':

        if request.form.get("avatar"):
            file_name = "." + request.form.get("avatar")

        animate = request.form.get("animate")
        avatar_original_img = cv2.imread(file_name)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
