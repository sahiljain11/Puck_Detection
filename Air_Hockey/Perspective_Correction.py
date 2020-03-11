import cv2
import imutils
import numpy as np
import glob


pts = np.array([(26, 86), (547, 0), (586, 306), (44, 301)])          #taken clockwise

rect = np.zeros((4, 2), dtype="float32")
s = pts.sum(axis=1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
(tl, tr, br, bl) = rect                                 # top left, top right, bottom right, bottom left

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
M = cv2.getPerspectiveTransform(rect, dst)

file_path = "playstationcam1"
vidcap = cv2.VideoCapture(file_path + ".mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
what = file_path + "_fixed.avi"
out = cv2.VideoWriter(what, fourcc, 20.0, (maxWidth, maxHeight))

while True:

    success, img = vidcap.read()
    if success:
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        cv2.imshow("image", img)
        cv2.imshow("warped", warped)
        out.write(warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

out.release()
cv2.destroyAllWindows()
