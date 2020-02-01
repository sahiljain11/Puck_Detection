import cv2
import numpy as np

#vidcap = cv2.VideoCapture('testing1.avi')
vidcap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

count = 0

def draw (image):
    #image, (x1, y1) (x2, y2), (b, g, r), width
    cv2.line(image, (0, 0), (150, 150), (255, 0, 0) , 15)

    #image, (top left x1, y1), (bottom right x2, y2), (b, g, r), width
    #-1 denotes a filled in object
    cv2.rectangle(image, (15, 25), (200, 150), (0, 0, 255), 5)

    #image, (center coords x, y), radius, (b, g, r), width
    cv2.circle(image, (100, 63), 55, (0, 255, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Testing!', (0, 130), font, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

def image_operations (image):

    #how to explictly calling ROI (region of image)
    image[0:120, 0:120] = [255, 255, 255]

while True:

    success, image = vidcap.read()
    count += 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #out.write(image)       #For saving videos for data later

    #draw(image)
    #draw(gray)

    ret, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('image', image)
    cv2.imshow('gray', mask)

    #cv2.imwrite("frame%d.jpg" % count, image)

    #im = cv2.imread("frame%d.jpg" % count, cv2.IMREAD_GRAYSCALE)

    #detector = cv2.SimpleBlobDetector()
    #o = cv2.ORB_create()
    #keypoints = o.detect(im, None)

    ##Detect orangeish colors
    #im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (250, 190, 88))

    #cv2.imshow("Keypoints", im_with_keypoints)

    #if (count % 10 == 0):
    #    print('Finished with #%d' % count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
out.release()
cv2.destroyAllWindows()
