import cv2
import numpy as np
import time
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--stream", type=int, default=0, help="stream video from videostream. denote capture #")
ap.add_argument("-w", "--write", type=str, default="output.avi", help="write video stream to file")
ap.add_argument("-r", "--read", type=str, default="", help="read from existing video file")

args = vars(ap.parse_args())

if (args.get('read') != ""):
    vidcap = cv2.VideoCapture(args.get('read'))
else:
    vidcap = cv2.VideoCapture(args.get("stream"))

if (args.get('write') != ""):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

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
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([120, 255, 255])


def image_operations(image):
    # hue, saturation, and value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(image, image, mask=hsv_mask)

    median = cv2.medianBlur(res, 15)

    kernel = np.ones((15, 15), np.float32) / 255;
    smoothed = cv2.filter2D(median, -1, kernel)

    # cv2.imshow("median", median)
    # cv2.imshow("smoothed", smoothed)

    return smoothed


def different_operations(image):
    blur = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # actually gets the puck
    mask = cv2.erode(mask, None, iterations=2)  # removes any excess blobs that may be detected
    mask = cv2.dilate(mask, None, iterations=2)  # also removes any excess blobs that may be detected

    cv2.imshow("mask", mask)

    return mask


def detection(original, image):
    inverse_im = cv2.bitwise_not(image)

    keypoints = detector.detect(inverse_im)

    print("-----------------------------------")
    print("num of blobs: ", len(keypoints))
    for point in keypoints:
        print(point.pt[0], " ", point.pt[1])
    print("-----------------------------------")

    im_with_keypoints = cv2.drawKeypoints(original, keypoints, np.array([]),
                                          (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


while True:

    file = open("debug.txt", "+w")
    success, image = vidcap.read()
    count += 1
    if success:
        # out.write(image)       #For saving videos for data later

        image = imutils.resize(image, width=600)  # increases FPS
        # new_image = image_operations(image)
        new_image = different_operations(image)

        blobs = detection(image, new_image)
        cv2.imshow("blobs", blobs)

        # file.write("--------------------------------\n")
        # file.write(str(new_image) + "\n")
        # file.write("--------------------------------\n")

        # cv2.imshow('image', image)
        # cv2.imshow('new_image', new_image)

        # cv2.imwrite("frame%d.jpg" % count, image)

        # im = cv2.imread("frame%d.jpg" % count, cv2.IMREAD_GRAYSCALE)

    # if (count % 10 == 0):
    #    print('Finished with #%d' % count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        file.close()
        break

vidcap.release()
# out.release()
cv2.destroyAllWindows()
