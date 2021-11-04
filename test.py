import imutils
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt
import numpy as np
import dlib
import cv2

import app

global detector, predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')


def transform_avatar(avatar_img, avatar_points, src_img, src_points):
    avatar_new_face = np.zeros(src_img.shape, np.uint8)
    avatar_img_copy = np.copy(avatar_img)

    for i, indices in enumerate(app.gen_triangles_indices(src_points)):
        avatar_triangle_points = app.gen_xy_from_indices(avatar_points, indices)
        src_triangle_points = app.gen_xy_from_indices(src_points, indices)

        avatar_triangle, avatar_img_cropped, _ = app.gen_cropped_triangle(avatar_img, avatar_triangle_points)
        _, avatar_img_cropped_copy, _ = app.gen_cropped_triangle(avatar_img_copy, avatar_triangle_points)
        src_triangle, src_img_cropped, src_b_rect = app.gen_cropped_triangle(src_img, src_triangle_points)

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

    return avatar_new_face, avatar_img_copy


# Funkcja pomocnicza do rysowania trojkatow na zdjeciach
def draw(img, triangle_list):
    t = triangle_list
    cv2.line(img, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(img, (int(t[2]), int(t[3])), (int(t[4]), int(t[5])), (255, 255, 255), 1, cv2.LINE_AA, 0)
    cv2.line(img, (int(t[4]), int(t[5])), (int(t[0]), int(t[1])), (255, 255, 255), 1, cv2.LINE_AA, 0)


# Funkcja wylicza trojkaty, znajduje odpowiedniki w dst_img i rysuje je na zdjeciach
def draw_triangles(src_img, src_points, dst_img, dst_points):
    for indices in app.gen_triangles_indices(src_points):
        src_triangle_points = app.gen_xy_from_indices(src_points, indices)
        dst_triangle_points = app.gen_xy_from_indices(dst_points, indices)

        t_s = [val for sublist in src_triangle_points for val in sublist]
        t_d = [val for sublist in dst_triangle_points for val in sublist]
        draw(src_img, t_s)
        draw(dst_img, t_d)


def generate():
    src_img = cv2.imread("./static/1.png")
    avatar_img = cv2.imread("./static/elsa.jpg")

    # src_img = imutils.resize(src_img, height=500)
    # avatar_img = imutils.resize(avatar_img, height=500)

    src_without_landmarks, src_points = app.gen_landmarks(src_img, 0)
    avatar_without_landmarks, avatar_points = app.gen_landmarks(avatar_img, 0)

    shape = avatar_without_landmarks.shape

    # z avatara robimy shape src
    face, avatar_without_face = transform_avatar(avatar_without_landmarks, avatar_points, src_without_landmarks,
                                                 src_points)

    plt.figure()
    cv2.imshow("Face", face)

    plt.figure()
    cv2.imshow("Avatar without face ", src_without_landmarks)

    transform = PiecewiseAffineTransform()
    transform.estimate(avatar_points[0:27], src_points[0:27])
    face = warp(face, transform, output_shape=shape, order=1, mode='wrap')
    mask = warp(src_without_landmarks, transform, output_shape=shape, order=1, mode='wrap')

    face = cv2.normalize(face, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    plt.figure()
    cv2.imshow("Mask before applying", mask)
    print("Mask type ", mask.dtype, "\nMask shape ", mask.shape)
    print("Avatar type ", avatar_without_landmarks.dtype, "\nAvatar shape ", avatar_without_landmarks.shape)

    img_without_face = avatar_without_landmarks & mask

    plt.figure()
    cv2.imshow("Avatar with mask", img_without_face)

    img_with_face = img_without_face + face

    plt.figure()
    cv2.imshow("Result", img_with_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


generate()
