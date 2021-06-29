import cv2
import argparse
import numpy as np



# Parsing the arguments
def ArgParse():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("ImagePath", 
                    help="Path of the image file.")

    args = vars(ap.parse_args())            # Converting it to dictionary.
    
    return args


def DetectAruco(Image):
    # Aruco detection params
    Parameters =  cv2.aruco.DetectorParameters_create()					# Creating parameters object.
    ArucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)				# Specifying auco dictionary used.

    # Detecting cv2.aruco markers.
    Corners, IDs, _ = cv2.aruco.detectMarkers(Image, ArucoDict, parameters=Parameters)

    return Corners, IDs


def GetBoxCorners(ArucoCorners, ArucoIDs):
    ArucoIDs = [i[0] for i in ArucoIDs]

    BoxCorners = []
    for i in range(4):
        BoxCorners.append([int(ArucoCorners[ArucoIDs.index(i)][0][i][0]),
                           int(ArucoCorners[ArucoIDs.index(i)][0][i][1])])

    return BoxCorners


def GetBox(Image, NumOfMarkers=4):
    # Converting image to grayscale if it is not 
    if len(Image.shape) != 2 and Image.shape[-1] != 1:
        if Image.shape[-1] == 3:
            Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        else:
            Image = cv2.cvtColor(Image, cv2.COLOR_BGRA2GRAY)
        
    # Detecting Aruco markers
    ArucoCorners, ArucoIDs = DetectAruco(Image)

    if ArucoIDs is None or len(ArucoIDs) != NumOfMarkers:
        return None

    # Getting Box Corners
    BoxCorners = GetBoxCorners(ArucoCorners, ArucoIDs)

    return BoxCorners


if __name__ == "__main__":
    args = ArgParse()

    GrayImage = cv2.imread(args["ImagePath"], cv2.IMREAD_GRAYSCALE)

    BoxCorners = GetBox(GrayImage)

    if BoxCorners is None:
        print("All 4 markers not found.")
        exit()

    img = cv2.cvtColor(GrayImage, cv2.COLOR_GRAY2BGR)
    for c in BoxCorners:
        img[c[1]][c[0]] = [0, 0, 255]

    from matplotlib import pyplot as p
    p.imshow(img)
    p.show()

