import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np

import app
from skimage.transform import PiecewiseAffineTransform, warp

global out, detector, predictor, avatar_shape, camera_shape

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')


def transform_simple_version(src_img, src_points, transform_table, triangles_indices, dst_shapes,
                             dst_triangles_cropped):
    for i, indices in enumerate(triangles_indices):
        src_triangle = app.gen_xy_from_indices(src_points, indices)

        src_triangle_cropped, src_img_cropped = app.gen_cropped_triangle(src_img, src_triangle)

        dst_img_warped = cv2.warpAffine(src_img_cropped, transform_table[i],
                                        (src_img_cropped.shape[1], src_img_cropped.shape[0]), None,
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros(src_img_cropped.shape, dtype=np.uint8)

        cv2.fillConvexPoly(mask, np.int32(src_triangle_cropped), (1.0, 1.0, 1.0), 16, 0);

        src_img_cropped *= 1 - mask

        src_img_cropped += dst_img_warped * mask


def transform_avatar(src_img, src_points, dst_img, dst_points):
    dst_img_new_face = np.zeros(dst_img.shape, np.uint8)
    src_img_copy = np.copy(src_img)
    # Funkcja gen_triangles indices generuje nam trojkaty ze wszystkich punktow w src_points i zwraca indeksy odpowiednich wartosci
    for indices in app.gen_triangles_indices(src_points):
        # Na podstawie trojkatow z src_points musimy też uzykac odpowienie trojkaty z dst_points
        # Funkcja gen_xy_from_indices generuje wspolrzedne z indeksow
        src_triangle_points = app.gen_xy_from_indices(src_points, indices)
        dst_triangle_points = app.gen_xy_from_indices(dst_points, indices)

        # Generujemy przyciete zdjecie w ktorym znajduje sie trojkat oraz sam trojkat
        src_triangle, src_img_cropped, src_b_rect = app.gen_cropped_triangle(src_img, src_triangle_points)

        _, src_img_cropped_2, _ = app.gen_cropped_triangle(src_img_copy, src_triangle_points)



        dst_triangle, dst_img_cropped, dst_b_rect = app.gen_cropped_triangle(dst_img, dst_triangle_points)

        x, y, w, h = dst_b_rect

        points_src = np.float32(src_triangle)
        points_dst = np.float32(dst_triangle)
        transform_matrix = cv2.getAffineTransform(points_src, points_dst)

        dst_img_warped = cv2.warpAffine(src_img_cropped, transform_matrix,
                                        (dst_img_cropped.shape[1], dst_img_cropped.shape[0]), None,
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros(dst_img_cropped.shape, dtype=np.uint8)
        src_mask = np.zeros(src_img_cropped.shape, dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, np.int32(dst_triangle), (1.0, 1.0, 1.0), 16, 0);
        src_mask = cv2.fillConvexPoly(src_mask, np.int32(src_triangle), (1.0, 1.0, 1.0), 16, 0);

        # W tym momencie zachowujemy na zdjeciu tylko te elementy poza dst_triangle
        dst_img_cropped *= 1 - mask


        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        cv2.fillConvexPoly(cropped_tr2_mask, np.int32(dst_triangle), 255)

        # if i == 3:
        #     plt.figure()
        #     cv2.imshow("Mask", mask)
        #
        #     plt.figure()
        #     cv2.imshow("Src", src_img_cropped)
        #
        #     plt.figure()
        #     cv2.imshow("Dst with black triangle", dst_img_cropped)
        #
        #     plt.figure()
        #     cv2.imshow("Warped", dst_img_warped)

        # W tym momencie dodajemy sam trojkat!
        #dst_img_cropped += dst_img_warped * mask

        # if i ==3:
        #     plt.figure()
        #     cv2.imshow("Dst after warped addition", dst_img_cropped)

        #warped_triangle = cv2.warpAffine(src_img_cropped, transform_matrix, (w, h))
        dst_img_warped = cv2.bitwise_and(dst_img_warped, dst_img_warped, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = dst_img_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        dst_img_warped = cv2.bitwise_and(dst_img_warped, dst_img_warped, mask=mask_triangles_designed)

        img2_new_face_rect_area += dst_img_warped
        dst_img_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

        src_img_cropped_2 *= 1 - src_mask

    return dst_img_new_face, src_img_copy


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


def transform_new(src_img, src_points, dst_img, dst_points):
    for indices in app.gen_triangles_indices(src_points):
        # print(indices)
        src_triangle_points = app.gen_xy_from_indices(src_points, indices)
        dst_triangle_points = app.gen_xy_from_indices(dst_points, indices)

        src_triangle_image, src_img_cropped = app.gen_cropped_triangle(src_img, src_triangle_points)
        dst_triangle_image, dst_img_cropped = app.gen_cropped_triangle(dst_img, dst_triangle_points)

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


def calculate_transform(src_img, dst_img, src_points, dst_points):
    transform_table = []
    triangles_indices = []
    dst_img_shape = []
    dst_triangles_cropped = []

    for i, indices in enumerate(app.gen_triangles_indices(src_points)):
        src_triangle = src_points[indices]
        dst_triangle = dst_points[indices]

        src_triangle_cropped, src_img_cropped = app.gen_cropped_triangle(src_img, src_triangle)
        dst_triangle_cropped, dst_img_cropped = app.gen_cropped_triangle(dst_img, dst_triangle)

        transform = cv2.getAffineTransform(np.float32(src_triangle_cropped), np.float32(dst_triangle_cropped))
        print("Indices: " + str(indices) + "Iteration: " + str(i))
        print("Transform: ", transform)
        transform_table.append(transform)
        triangles_indices.append(indices)
        dst_img_shape.append(dst_img_cropped.shape)
        dst_triangles_cropped.append(dst_triangle_cropped)

    return transform_table, triangles_indices, dst_img_shape, dst_triangles_cropped


def generate():
    src_img = cv2.imread("./static/1.png")
    test_img = cv2.imread("./static/elsa.jpg")
    src_img_copy = np.copy(src_img)

    src_img_without, src_points = app.gen_landmarks(src_img, 0)
    dst_img_without, dst_points = app.gen_landmarks(test_img, 0)

    # transform_table, triangles_indices, dst_shapes, dst_triangles_cropped = calculate_transform(src_img, dst_img, np.array(src_points), np.array(dst_points))
    #
    # test_img, test_points = app.gen_landmarks(test_img, 0)
    # transform_simple_version(test_img, np.array(test_points), transform_table, triangles_indices, dst_shapes, dst_triangles_cropped)

    # transform_avatar(test_img, dst_points, src_img, src_points)

    draw_triangles(src_img_copy, src_points, test_img, dst_points)

    face, elsa_without_face = transform_avatar(dst_img_without, dst_points, src_img_without, src_points)

    plt.figure()
    cv2.imshow("Face", face)

    #draw_triangles(src_img, src_points, test_img, dst_points)

    plt.figure()
    cv2.imshow("Without face", src_img_without)

    src_img_without += face

    plt.figure()
    cv2.imshow("Elsa without face", elsa_without_face)

    tform = PiecewiseAffineTransform()
    tform.estimate(dst_points[0:27], src_points[0:27])
    out = warp(face, tform, output_shape=elsa_without_face.shape)

    plt.figure()
    cv2.imshow("XD", out)

    plt.figure()
    cv2.imshow("XDDD", elsa_without_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


generate()
