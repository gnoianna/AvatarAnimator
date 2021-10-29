import cv2

coo_array = []
counter = 0
file_name = './static/tiger_2.jpg'


def click_event(event, x, y, flags, params):
    global counter

    if event == cv2.EVENT_LBUTTONDOWN:

        if counter <= 67:
            coo_array.append([x, y])
            cv2.putText(img, '.' + str(counter), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)
        else:
            print("No more landmarks to choose")
            pass
        cv2.imshow(file_name, img)
        counter += 1

    if event == cv2.EVENT_RBUTTONDOWN:
        print("[", end='')
        for index, coo in enumerate(coo_array):
            if index == len(coo_array) - 1:
                print(str(coo) + "]")
            else:
                print(str(coo) + ",")


if __name__ == "__main__":
    img = cv2.imread(file_name, 1)
    cv2.imshow(file_name, img)

    cv2.setMouseCallback(file_name, click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
