'''
pip install opencv-python
pip install cellpose
pip install scikit-image
pip install mxnet
pip uninstall mxnet-mkl -y
pip install mxnet-cu101 
'''

import cv2
import numpy as np
import cellpose
from cellpose import utils
from cellpose import core
from cellpose import models
import time as t
from pandas import DataFrame, Series


import time, os, sys
from urllib.parse import urlparse
#import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
#%matplotlib inline
mpl.rcParams['figure.dpi'] = 300

import mxnet as mx


t1 = t.time()

'''########################################################################################################
InputImagePath = "Wheat.jpg"
CSV_name = "ContoursFound.csv"      # If changed, change this file name in .gitignore too.

# For full sized image -> diameter = 50
# For 0.25 times image size -> diameter = 30
Rescale_Factor = 0.25		# Image will be resized by this value.
Diameter = 20
########################################################################################################

'''

def NumberGrain(BlueSheet, Contours, StartingIndex=0):
    Font = cv2.FONT_HERSHEY_SIMPLEX
    FontScale = 0.4
    Thickness = 1
    Colour = (0, 0, 255)

    BlueSheetCopy1 = BlueSheet.copy()
    cv2.drawContours(BlueSheetCopy1, Contours, -1, (0, 255, 0), 1)

    for i in range(len(Contours)):
        Contour = Contours[i]
        x, y, w, h = cv2.boundingRect(Contour)
        BottomLeftPoint = ( x, y )
        BlueSheetCopy1 = cv2.putText(BlueSheetCopy1, str(cv2.contourArea(Contour)), BottomLeftPoint, Font, FontScale, Colour, Thickness, cv2.LINE_AA)


    return BlueSheetCopy1


def ApplyCellpose(Model, Image, Diameter):
	
	masks, flows, styles, diams = Model.eval(Image, diameter=Diameter, channels=[0, 0], do_3D=False)
	
	outlines = cellpose.utils.outlines_list(masks)

	#Numbered = NumberGrain(Image, outlines, StartingIndex=1)
	#cv2.imwrite("Numbered.jpg", Numbered)
 	
	return outlines


def FindParams(Contour):
    ParamsList = []

    # Area
    Area = cv2.contourArea(Contour)
    ParamsList.append(Area)

    # Width and Length
    Rect = cv2.minAreaRect(Contour)
    ParamsList.append(Rect[1][0])       # Width
    ParamsList.append(Rect[1][1])       # Length
    ParamsList.append(Rect[1][0] / Rect[1][1])       # Width/Length

    # Circularity
    Perimeter = cv2.arcLength(Contour, True)
    Circularity = 4 * np.pi * (Area / (Perimeter * Perimeter))
    ParamsList.append(Circularity)

    return ParamsList


def py_main(InputImagePath, CSV_name, Rescale_Factor, Diameter):

    # Global variables shifted here
    use_GPU = core.use_gpu()
    print('GPU activated? %d'%use_GPU)

    # model_type='cyto' or model_type='nuclei'
    Model = models.Cellpose(gpu=use_GPU, model_type='cyto', net_avg=False)


    # Reading and manipulating image
    Image = cv2.imread(InputImagePath)
    print("Image shape: {}".format(Image.shape[:2]))
    
    Image = cv2.resize(Image, (0, 0), fx=Rescale_Factor, fy=Rescale_Factor)
    Image = cv2.fastNlMeansDenoisingColored(Image, None, 10, 10, 7, 21)
    
    # Applying Cellpose
    Contours = ApplyCellpose(Model, Image, Diameter)
    print("Number of Contours: " + str(len(Contours)))
    
    # Storing data in csv
    Contour_Dict = dict()
    
    for i in range(len(Contours)):
        Contour_Dict[str(i)] = FindParams(Contours[i])
    
    df = DataFrame(dict([ (k,Series(v)) for k,v in Contour_Dict.items() ]))
    df = df.T
    df.to_csv(CSV_name, header=False, index=False) 
    
    print(t.time() - t1)
    
    return Contour_Dict


#py_main("Wheat.jpg", "ContoursFound.csv", 0.25, 20)
