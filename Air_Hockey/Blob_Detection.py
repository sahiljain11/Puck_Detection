import cv2
import numpy as np

#vidcap = cv2.VideoCapture('testing1.avi')
vidcap = cv2.VideoCapture(-1)
success, image = vidcap.read()
count = 0

while True:

    cv2.imshow('image', image)

    #cv2.imwrite("frame%d.jpg" % count, image)
    #
    #im = cv2.imread("frame%d.jpg" % count, cv2.IMREAD_GRAYSCALE)

    #detector = cv2.SimpleBlobDetector()
    #o = cv2.ORB_create()
    #keypoints = o.detect(im, None)
    #
    ##Detect orangeish colors
    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (250, 190, 88))

    #cv2.imshow("Keypoints", im_with_keypoints)

    #if (count % 10 == 0):
    #    print('Finished with #%d' % count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    
    success, image = vidcap.read()
    count += 1

vidcap.release()
