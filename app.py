import cv2
import dlib
import imutils
import numpy as np
from flask import Flask, render_template, Response, request
from imutils import face_utils
from skimage.transform import PiecewiseAffineTransform, warp

global out, detector, predictor, avatar_temp_image, animate

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)
file_name = "./static/elsa.jpg"
avatar_temp_image = cv2.imread(file_name)


def all_inside(avatar, points, src_img, src_points):
    global avatar_temp_image
    print("all_inside")
    avatar_new_face = np.zeros(src_img.shape, np.uint8)
    avatar_img_copy = np.copy(avatar)
    avatar_img_copy_copy = np.copy(avatar)
    src_img_copy = np.copy(src_img)

    for i, indices in enumerate(gen_triangles_indices(src_points)):
        avatar_triangle_points = gen_xy_from_indices(points, indices)
        src_triangle_points = gen_xy_from_indices(src_points, indices)

        avatar_triangle, avatar_img_cropped, _ = gen_cropped_triangle(avatar_img_copy_copy, avatar_triangle_points)
        _, avatar_img_cropped_copy, _ = gen_cropped_triangle(avatar_img_copy, avatar_triangle_points)
        src_triangle, src_img_cropped, src_b_rect = gen_cropped_triangle(src_img_copy, src_triangle_points)

        x, y, w, h = src_b_rect

        transform_matrix = cv2.getAffineTransform(np.float32(avatar_triangle), np.float32(src_triangle))

        avatar_t_warped = cv2.warpAffine(avatar_img_cropped, transform_matrix,
                                         (src_img_cropped.shape[1], src_img_cropped.shape[0]), None,
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros(src_img_cropped.shape, dtype=np.uint8)
        avatar_mask = np.zeros(avatar_img_cropped.shape, dtype=np.uint8)

        mask = cv2.fillConvexPoly(mask, np.int32(src_triangle), (1.0, 1.0, 1.0), 16, 0)
        avatar_mask = cv2.fillConvexPoly(avatar_mask, np.int32(avatar_triangle), (1.0, 1.0, 1.0), 16, 0)

        src_img_cropped *= 1 - mask
        avatar_img_cropped_copy *= 1 - avatar_mask
        avatar_t_warped *= mask

        new_face_rect_area_gray = cv2.cvtColor(avatar_new_face[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 0, 255, cv2.THRESH_BINARY_INV)
        avatar_t_warped = cv2.bitwise_and(avatar_t_warped, avatar_t_warped, mask=mask_triangles_designed)

        avatar_new_face[y: y + h, x: x + w] += avatar_t_warped

    transform = PiecewiseAffineTransform()
    transform.estimate(points[0:27], src_points[0:27])
    face = warp(avatar_new_face, transform, output_shape=avatar_img_copy_copy.shape, order=1, mode='wrap')

    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    _, face_mask = cv2.threshold(face_gray, 1, 255, cv2.THRESH_BINARY_INV)
    avatar_kkk = cv2.bitwise_and(avatar, avatar, mask=face_mask)

    avatar_kkk += face

    avatar_temp_image = avatar_kkk


def gen_frames_with_landmarks():
    global out, detector, predictor, avatar_temp_image, animate
    print("gen_frames_with_landmarks")

    avatar = cv2.imread(file_name)


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
                    animate = 0
                    avatar, avatar_shape = gen_landmarks(avatar, 0)
                    temp_avatar = np.copy(avatar)
                    all_inside(temp_avatar, avatar_shape, frame, camera_shape)

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
    print("gen_avatar_stream")

    while True:
        try:
            _, encoded_avatar = cv2.imencode(".jpg", avatar_temp_image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_avatar) + b'\r\n')
        except Exception as e:
            print(e)

        else:
            pass


def without_landmarks(image):
    return np.copy(image)


def gen_landmarks(image, if_with_landmarks):
    global detector, predictor

    without_landmarks_image = without_landmarks(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        avatar_shape = predictor(gray, rect)
        avatar_shape = face_utils.shape_to_np(avatar_shape)

        for i, (x, y) in enumerate(avatar_shape):
            image = cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    if if_with_landmarks:
        return image, avatar_shape
    else:
        return without_landmarks_image, avatar_shape


def gen_triangles_indices(points):
    b_rect = (*points.min(axis=0), *points.max(axis=0))

    subdiv = cv2.Subdiv2D(b_rect)
    points_list = list(points)
    subdiv.insert(points_list)
    triangle_list = subdiv.getTriangleList()

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


def transform(src_img, src_points, dst_img, dst_points):
    for indices in gen_triangles_indices(src_points):
        src_triangle_points = gen_xy_from_indices(src_points, indices)
        dst_triangle_points = gen_xy_from_indices(dst_points, indices)

        src_triangle_image, src_img_cropped, _ = gen_cropped_triangle(src_img, src_triangle_points)
        dst_triangle_image, dst_img_cropped, _ = gen_cropped_triangle(dst_img, dst_triangle_points)

        points_src = np.float32(src_triangle_image)
        points_dst = np.float32(dst_triangle_image)
        transform_matrix = cv2.getAffineTransform(points_src, points_dst)

        transformed_triangle = cv2.warpAffine(src_img_cropped, transform_matrix,
                                              (dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None,
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros(dst_img_cropped.shape, dtype=np.uint8)

        mask = cv2.fillConvexPoly(mask, np.int32(dst_triangle_image), (1.0, 1.0, 1.0), 16, 0);

        mask_opp = 1 - mask
        dst_img_cropped *= mask_opp
        dst_img_cropped += transformed_triangle * mask


@app.route('/')
def index():
    global animate
    print("index")
    animate = 0
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
        animate = request.form.get("animate")
        file = request.form.get("avatar")
        print(file)
        #avatar_temp_image = cv2.imread(file_name)


    return render_template('index.html')


if __name__ == '__main__':
    app.run()
