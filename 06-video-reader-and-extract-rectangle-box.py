import cv2

cap = cv2.VideoCapture("highway.mp4")

# object detection from stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2()

# change the object detection algorithm
# but we still get losts of wrong greens, tune the threshold higher
# varThreshold = 5, 40
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    # To make it work, we should choose the region of interesting area(roi)
    # and we will focus only on this specific part of the road
    # and we only focus the vehicles on this road.
    # So that we can track them correctly.

    # Let's define an roi which will be region of interest.
    # Extract Region of interest
    height, width, _ = frame.shape
    print(height, width)  # 720 * 1280
    # what is the best roi? Compare it with the original picture.
    roi = frame[340:720, 500:800]

    # Let's rmeove all the small elements to remove the small elements.
    mask = object_detector.apply(roi)
    # contour: an outline especially of a curving or irregular figure
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:  # less than 100 pixels
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            # We're going to extract the rectangle so the box that surrounds each object detected.
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y+h), (0, 255, 0), 3)

        # else: Skip them less than 100 pixels.

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
