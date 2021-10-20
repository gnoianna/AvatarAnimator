import cv2
import dlib
import imutils
import numpy as np
from flask import Flask, render_template, Response, request, send_file
from imutils import face_utils

global out, detector, predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)

def gen_frames_with_landmarks():
    global out, detector, predictor

    src_img = cv2.imread("./static/woman_ch.jpg")
    image, avatar_shape = gen_landmarks(src_img, 0)

    while True:
        success, frame = camera.read()
        frame = imutils.resize(frame)
        if success:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            rects = [item for item in rects]
            if rects:
                camera_shape = predictor(gray, rects[0])
                camera_shape = face_utils.shape_to_np(camera_shape)

                for (x, y) in camera_shape:
                    frame = cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # print("Camera shape:\n", camera_shape)

                transform(image, avatar_shape, frame, camera_shape)

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
    img = cv2.imread("./static/woman_ch_c.jpg")
    new_img, _ = gen_landmarks(img, 1)

    try:
        ret, buffer = cv2.imencode('.jpg', new_img)
        new_img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + new_img + b'\r\n')
    except Exception as e:
        print(e)

    else:
        pass


def without_landmarks(image):
    return image.copy()


def gen_landmarks(image, if_with_landmarks):
    global detector, predictor

    without_landmarks_image = without_landmarks(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        avatar_shape = predictor(gray, rect)
        avatar_shape = face_utils.shape_to_np(avatar_shape)

        for (x, y) in avatar_shape:
            image = cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # print("Avatar shape:\n", avatar_shape)
    if if_with_landmarks:
        return image, avatar_shape
    else:
        return without_landmarks_image, avatar_shape


def get_triangulation_indices(points):
    bounding_rect = (*points.min(axis=0), *points.max(axis=0))

    subdiv = cv2.Subdiv2D(bounding_rect)
    subdiv.insert(list(points))

    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():
        yield [(points == point).all(axis=1).nonzero()[0][0] for point in [(x1, y1), (x2, y2), (x3, y3)]]


def crop_to_triangle(img, triangle):
    bounding_rect = cv2.boundingRect(triangle)

    img_cropped = img[bounding_rect[1]:bounding_rect[1] + bounding_rect[3],
                  bounding_rect[0]:bounding_rect[0] + bounding_rect[2]]

    triangle_cropped = [(point[0] - bounding_rect[0], point[1] - bounding_rect[1]) for point in triangle]
    return triangle_cropped, img_cropped


def get_triangles_from_indices(points, indices):
    triangles = [points[single_index] for single_index in indices]

    return np.array(triangles)


def transform(src_img, src_points, dst_img, dst_points):
    for indices in get_triangulation_indices(src_points):
        src_triangle = get_triangles_from_indices(src_points, indices)
        dst_triangle = get_triangles_from_indices(dst_points, indices)

        src_triangle_cropped, src_img_cropped = crop_to_triangle(src_img, src_triangle)
        dst_triangle_cropped, dst_img_cropped = crop_to_triangle(dst_img, dst_triangle)

        transform_matrix = cv2.getAffineTransform(np.float32(src_triangle_cropped), np.float32(dst_triangle_cropped))

        dst_img_warped = cv2.warpAffine(src_img_cropped, transform_matrix,
                                        (dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None,
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros(dst_img_cropped.shape, dtype=np.uint8)

        mask = cv2.fillConvexPoly(mask, np.int32(dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0);

        dst_img_cropped *= 1 - mask
        dst_img_cropped += dst_img_warped * mask

        # seamlessclone = cv2.seamlessClone(dst_img_cropped, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)


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
