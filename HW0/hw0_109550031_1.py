import cv2 as cv

file = open('bounding_box.txt','r')
img = cv.imread('image.png')
                 
for line in file.readlines():
    num = line.split(" ")
    x1 = int(num[0])
    y1 = int(num[1])
    x2 = int(num[2])
    y2 = int(num[3])
    cv.rectangle(img, (x1,y1), (x2,y2), (0,0,255), thickness = 3)
    
cv.imwrite('hw0_109550031_1.png', img)
cv.imshow('My Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
