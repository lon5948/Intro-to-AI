import os
import cv2
import numpy
def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    dataset = []
    car = dataPath+'/car'
    allfiles = os.listdir(car)
    
    for file in allfiles:
        img = cv2.imread(car+'/'+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(36,16))
        label = 1
        tuples = (img,label)
        dataset.append(tuples)
        
    noncar = dataPath+'/non-car'
    allfiles = os.listdir(noncar)
    
    for file in allfiles:
        img = cv2.imread(noncar+'/'+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(36,16))
        label = 0
        tuples = (img,label)
        dataset.append(tuples)
    
    return dataset
