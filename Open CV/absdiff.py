import cv2 as cv
import numpy as np

video = cv.VideoCapture('video.mp4')

while(1):
    ret, frame1 = video.read()
    ret, frame2 = video.read()
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    blur1 = cv.GaussianBlur(gray1,(5,5),0)
    blur2 = cv.GaussianBlur(gray2,(5,5),0)
    result = cv.absdiff(blur1,blur2)
    result = cv.cvtColor(result,cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(result,cv.COLOR_BGR2HSV)
    lower_red = np.array([0,0,0])
    upper_red = np.array([0,50,50])
    mask=cv.inRange(hsv,lower_red,upper_red)
    result[mask>0]=(0,0,0)
    result[mask<=0]=(0,255,0)
    toge = np.hstack((frame2,result))
    toge = cv.resize(toge, (2000, 500))
    cv.imshow('toge',toge)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.imwrite('hw0_109550031_2.png', toge)
        break

video.release()
cv.destroyAllWindows()