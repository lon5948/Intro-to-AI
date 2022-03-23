import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from adaboost import Adaboost
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    coordinate = {}
    coordinate['x1'] = []
    coordinate['y1'] = []
    coordinate['x2'] = []
    coordinate['y2'] = []
    coordinate['x3'] = []
    coordinate['y3'] = []
    coordinate['x4'] = []
    coordinate['y4'] = []
    
    file = open(dataPath,'r')
    parking_space = int(file.readline())
    
    for line in file.readlines():
        num = line.split(" ") 
        coordinate['x1'].append(int(num[0]))
        coordinate['y1'].append(int(num[1]))
        coordinate['x2'].append(int(num[2]))
        coordinate['y2'].append(int(num[3]))
        coordinate['x3'].append(int(num[4]))
        coordinate['y3'].append(int(num[5]))
        coordinate['x4'].append(int(num[6]))
        coordinate['y4'].append(int(num[7]))
            
    video = cv2.VideoCapture('data/detect/video.gif')
    f = open('Adaboost_pred.txt','w')
    
    while(1):
        ret, img = video.read()
        if ret == False:
            break
        for i in range(parking_space):
            x1 = coordinate['x1'][i]
            y1 = coordinate['y1'][i]
            x2 = coordinate['x2'][i]
            y2 = coordinate['y2'][i]
            x3 = coordinate['x3'][i]
            y3 = coordinate['y3'][i]
            x4 = coordinate['x4'][i]
            y4 = coordinate['y4'][i]
            crop_img = crop(x1,y1,x2,y2,x3,y3,x4,y4,img)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            crop_img = cv2.resize(crop_img,(36,16))
            label = clf.classify(crop_img)
            f.write(str(label))
            f.write(' ')
            if label == 1:
                points = np.array([ [x1,y1],[x2,y2],[x4,y4],[x3,y3]], np.int32)
                cv2.polylines(img,[points],True,(0,255,0),2)
        f.write('\n')
        cv2.imshow('img',img)
        cv2.waitKey(0)
        
    video.release()
    cv2.destroyAllWindows()
