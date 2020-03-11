import cv2
import numpy as np
import time
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", type=int, default=0, help="stream video from videostream. denote capture number (default=0)")
ap.add_argument("-w", "--write", type=str, default="output.avi", help="write video stream to file (default=\"output.avi\")")
ap.add_argument("-r", "--read", type=str, default="", help="read from existing video file")
ap.add_argument("-d", "--debug", help="debug by piping extra info into debug.txt")

args = vars(ap.parse_args())

if (args.get('read') != ""):
    vidcap = cv2.VideoCapture(args.get('read'))
else:
    vidcap = cv2.VideoCapture(args.get("stream"))

vidcap = cv2.VideoCapture("playstationcam1_fixed.avi")

if (args.get('write') != ""):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

time.sleep(2.0)     # Helps in loading up video processing
count = 0
blob_params = cv2.SimpleBlobDetector_Params()

# Filter by Color
blob_params.filterByColor = True
blob_params.filterByArea = False
blob_params.filterByConvexity = False
blob_params.filterByInertia = False
blob_params.filterByCircularity = False

detector = cv2.SimpleBlobDetector_create(blob_params)
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([120, 255, 255])

def image_operations(image):
    blur = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # actually gets the puck
    mask = cv2.erode(mask, None, iterations=3)  # removes any excess blobs that may be detected
    mask = cv2.dilate(mask, None, iterations=2)  # also removes any excess blobs that may be detected

    cv2.imshow("mask", mask)

    return mask


def detection(original, image):
    inverse_im = cv2.bitwise_not(image)

    keypoints = detector.detect(inverse_im)

    # print("-----------------------------------")
    # print("num of blobs: ", len(keypoints))
    # for point in keypoints:
    #     print(point.pt[0], " ", point.pt[1])
    # print("-----------------------------------")

    im_with_keypoints = cv2.drawKeypoints(original, keypoints, np.array([]),
                                          (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


while True:

    if args.get("debug") != "":
        file = open("debug.txt", "+w")

    success, image = vidcap.read()
    count += 1
    if success:

        if args.get("write") != "":
            out.write(image)       #For saving videos for data later

        image = imutils.resize(image, width=600)  # increases FPS
        new_image = image_operations(image)

        blobs = detection(image, new_image)
        cv2.imshow("blobs", blobs)

        # if (args.get("debug") != ""):
            # file.write("--------------------------------\n")
            # file.write(str(new_image) + "\n")
            # file.write("--------------------------------\n")

        # cv2.imwrite("frame%d.jpg" % count, image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        if (args.get("debug") != ""):
            file.close()
        break

if (args.get("write") != ""):
    out.release()

vidcap.release()
cv2.destroyAllWindows()
