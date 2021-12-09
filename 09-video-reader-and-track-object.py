import cv2
from tracker import *

# create tracker object
tracker = EuclideanDistTracker()
# How th tracker work?
# This tracker takes all the bounding boxes of the objects
# we need to save the bounding boxes of the object into one array
#

cap = cv2.VideoCapture("highway.mp4")

# object detection from stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2()

# change the object detection algorithm
# but we still get losts of wrong greens, tune the threshold higher
# varThreshold = 5, 40
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# looping frame after frame on the video
while True:
    ret, frame = cap.read()

    # To make it work, we should choose the region of interesting area(roi)
    # and we will focus only on this specific part of the road
    # and we only focus the vehicles on this road.
    # So that we can track them correctly.

    # Let's define an roi which will be region of interest.
    # Extract Region of interest
    height, width, _ = frame.shape
    # print(height, width)  # 720 * 1280
    # what is the best roi? Compare it with the original picture.
    roi = frame[340:720, 500:800]

    # Let's rmeove all the small elements to remove the small elements.
    mask = object_detector.apply(roi)
    # keep it in the mind that pixel are from 0 to 255.
    # 0 is completely black and 255 is completely white.
    # all the white from the shadow was removed.
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    # contour: an outline especially of a curving or irregular figure
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area <= 100:
            continue

        # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
        # We're going to extract the rectangle so the box that surrounds each object detected.
        # x: x position
        # y: y position
        # w: width
        # h: height
        x, y, w, h = cv2.boundingRect(cnt)
        # put all the bounding box into one array
        # print(x, y, w, h) # changed frequently
        detections.append([x, y, w, h])

    print(detections)
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    # [x,y,w,h, id]
    # [277, 226, 23, 67, 7]
    print(boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        # we want to put the text where on the roi
        # Put the id just following the object.
        # make the text of id higher than the object in blue.
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y+h), (0, 255, 0), 3)

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # key = cv2.waitKey(30)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
