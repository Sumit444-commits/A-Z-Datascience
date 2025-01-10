import numpy as np
import cv2 as cv

# # read the webcam using cv
# cap = cv.VideoCapture(0)
# if cap.isOpened():
#     ret, frame = cap.read()
# else:
#     ret = False
    
# while ret:
#     ret, frame = cap.read()
#     cv.imshow('frame', frame)
#     if cv.waitKey(25) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv.destroyAllWindows()


# read the webcam using cv
cap = cv.VideoCapture(0)

# change the cam into gray scale cam
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    cv.imshow('frame', gray)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()