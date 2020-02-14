import cv2
import numpy as np

vidcap = cv2.VideoCapture('test_vid.MOV')
#vidcap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

    #Not really too useful in this case but it's pretty good for clearing up an image
    #guassian = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 1)

    #hue, saturation, and value
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([50, 0, 0])
    upper_blue = np.array([120, 255, 255])

    hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(image, image, mask=hsv_mask)

    #kernel = np.ones((15, 15), np.float32) / 255;
    #smoothed = cv2.filter2D(res, -1, kernel)

    #clearer = cv2.GaussianBlur(res, (15, 15), 0)
    median = cv2.medianBlur(res, 15)

    cv2.imshow("median", median)
    #cv2.imshow("clearer", clearer)
    #cv2.imshow("smoothed", smoothed)
    return image

while True:

    success, image = vidcap.read()
    count += 1
    if success:
        #out.write(image)       #For saving videos for data later

        #draw(image)
        #draw(gray)

        new_image = image_operations(image)
        #cv2.imshow('image', image)
        #cv2.imshow('new_image', new_image)

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
#out.release()
cv2.destroyAllWindows()

# how to explictly calling ROI (region of image)
# image[0:120, 0:120] = [255, 255, 255]
