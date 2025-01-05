import numpy as np
import cv2

window_name = "Window"
interframe_wait_ms=10
# Capturing video through webcam
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Start a while loop
while 1:

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()# _, is used to skip the first value of list or tuple....see t.py for clarity

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for gray color and
    # define mask
    gray_lower = np.array([0, 0, 70], np.uint8)#remember BGR
    gray_upper = np.array([170, 10, 170], np.uint8)
    gray_mask = cv2.inRange(hsvFrame, gray_lower, gray_upper)

    # Set range for black color and
    # define mask
    black_lower = np.array([0, 0, 0], np.uint8)  # remember BGR
    black_upper = np.array([0, 0, 255], np.uint8)
    black_mask = cv2.inRange(hsvFrame, black_lower, black_upper)

    # Set range for white color and
    # define mask
    white_lower = np.array([0, 0, 230], np.uint8)  # remember BGR
    white_upper = np.array([70, 0, 255], np.uint8)
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper)

    # Set range for yellow color and
    # define mask
    yellow_lower = np.array([30, 50, 175], np.uint8)  # remember BGR
    yellow_upper = np.array([34,170,255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # Set range for red color and
    # define mask
    red_lower = np.array([175, 85, 175], np.uint8)
    red_upper = np.array([180, 255, 205], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    green_lower = np.array([50, 105, 85], np.uint8)
    green_upper = np.array([85, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 120, 120], np.uint8)
    blue_upper = np.array([102, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Set range for violet color and
    # define mask
    violet_lower = np.array([130, 60, 111], np.uint8)
    violet_upper = np.array([140, 255, 255], np.uint8)
    violet_mask = cv2.inRange(hsvFrame, violet_lower, violet_upper)

    # Set range for orange color and
    # define mask
    orange_lower = np.array([12, 180, 230], np.uint8)
    orange_upper = np.array([20, 220, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # Set range for brown color and
    # define mask
    brown_lower = np.array([0, 100, 10], np.uint8)
    brown_upper = np.array([10, 220, 110], np.uint8)
    brown_mask = cv2.inRange(hsvFrame, brown_lower, brown_upper)


    kernel = np.ones((5, 5), "uint8")

    # For gray color
    gray_mask = cv2.dilate(gray_mask, kernel)
    res_gray = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=gray_mask)

    # For black color
    black_mask = cv2.dilate(black_mask, kernel)
    res_black = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=black_mask)

    # For white color
    white_mask = cv2.dilate(white_mask, kernel)
    res_white = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=white_mask)

    # For yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernel)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=yellow_mask)
    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernel)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernel)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)

    # For violet color
    violet_mask = cv2.dilate(violet_mask, kernel)
    res_violet = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=violet_mask)

    # For orange color
    orange_mask = cv2.dilate(orange_mask, kernel)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=orange_mask)

    # For brown color
    brown_mask = cv2.dilate(brown_mask, kernel)
    res_brown = cv2.bitwise_and(imageFrame, imageFrame,
                                 mask=brown_mask)


    # Creating contour to track gray color
    contours, hierarchy = cv2.findContours(gray_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (128, 128, 128), 2)

            cv2.putText(imageFrame, "Gray", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (128, 128, 128))

    # Creating contour to track black color
    contours, hierarchy = cv2.findContours(black_mask,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                        (x + w, y + h),
                                        (0, 0, 0), 2)

            cv2.putText(imageFrame, "Black", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0))
    # Creating contour to track white color
    contours, hierarchy = cv2.findContours(white_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (255, 180, 255), 2)

            cv2.putText(imageFrame, "White", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (255,180, 255))
    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 255, 255), 2)

            cv2.putText(imageFrame, "Yellow", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255))





    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "Red", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 255, 0), 2)

            cv2.putText(imageFrame, "Green", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(imageFrame, "Blue", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0))

    # Creating contour to track violet color
    contours, hierarchy = cv2.findContours(violet_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (255, 0, 143), 2)

            cv2.putText(imageFrame, "Violet", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 0, 143))

    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(orange_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 140, 255), 2)

            cv2.putText(imageFrame, "Orange", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 140, 255))

    # Creating contour to track brown color
    contours, hierarchy = cv2.findContours(brown_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (140, 161, 210), 2)

            cv2.putText(imageFrame, "Brown/Skin", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (140, 161, 210))

    # Program Termination
    cv2.imshow(window_name, imageFrame)
    if cv2.waitKey(interframe_wait_ms) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break