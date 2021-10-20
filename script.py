import cv2

coo_array = []


def click_event(event, x, y, flags, params):
    counter = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        coo_array.append([x, y])
        cv2.putText(img, '.', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 2)
        cv2.imshow('./static/tiger_2.jpg', img)
        counter += 1

    if event == cv2.EVENT_RBUTTONDOWN:
        print("[", end='')
        for index, coo in enumerate(coo_array):
            if index == len(coo_array) - 1:
                print(str(coo) + "]")
            else:
                print(str(coo) + ",")


if __name__ == "__main__":
    img = cv2.imread('./static/tiger_2.jpg', 1)
    cv2.imshow('./static/tiger_2.jpg', img)

    cv2.setMouseCallback('./static/tiger_2.jpg', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
