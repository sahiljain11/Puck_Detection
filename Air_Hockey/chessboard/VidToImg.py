import cv2
import numpy as np
import imutils

vidcap = cv2.VideoCapture('chessboard2.MOV')
count = 0

while True:

    success, image = vidcap.read()

    if success:
        count += 1
        if (count % 5 == 0) and (count > 320):
            cv2.imwrite('frame%d.jpg' % count, image)

    if cv2.waitKey(1) & 0xFF == ord('q') or not success:
        break

vidcap.release()
cv2.destroyAllWindows()
