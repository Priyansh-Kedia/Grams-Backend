import cv2
import math
import numpy as np


def Angle(Pt1, Pt2):
	x1, y1, x2, y2 = Pt1[0], Pt1[1], Pt2[0], Pt2[1]

	LineAngle = math.degrees(math.atan( ((y2 - y1) / (x2 - x1)) ))

	return LineAngle


def DetectAruco(Image, NumOfAruco, DictionaryType = cv2.aruco.DICT_6X6_50):
    # Getting grayscale image
    if Image.shape[2] == 3:
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    elif Image.shape[2] == 4:
        Image = cv2.cvtColor(Image, cv2.COLOR_BGRA2GRAY)
        
    ArucoDict = cv2.aruco.Dictionary_get(DictionaryType)
    Parameters = cv2.aruco.DetectorParameters_create()
    
    Corners, IDs, RejectedImgPoints = cv2.aruco.detectMarkers(Image, ArucoDict, parameters=Parameters)

    if len(IDs) == NumOfAruco:
        return True, Corners, IDs
    return False, Corners, IDs


def GetArucoSideLengthInPixels(Corners):
    # List of side lengths in pixels (horizontal, verticle)
    SideLengths = []

    for ArucoCorners in Corners:
        ArucoCorners = ArucoCorners[0]

        # Determining if [0 - 1] if horizontal or verticle
        SideAngle = Angle(ArucoCorners[0], ArucoCorners[1])

        if abs(SideAngle) < 45.0:   # It is horizontal line
            SideLengths.append([(abs(ArucoCorners[0][0] - ArucoCorners[1][0]) + abs(ArucoCorners[2][0] - ArucoCorners[3][0])) / 2,
                                (abs(ArucoCorners[1][1] - ArucoCorners[2][1]) + abs(ArucoCorners[3][1] - ArucoCorners[0][1])) / 2] + [abs(SideAngle)])
        else:                       # It is verticle line
            SideLengths.append([(abs(ArucoCorners[1][0] - ArucoCorners[2][0]) + abs(ArucoCorners[3][0] - ArucoCorners[0][0])) / 2,
                                (abs(ArucoCorners[0][1] - ArucoCorners[1][1]) + abs(ArucoCorners[2][1] - ArucoCorners[3][1])) / 2] + [90.0 - abs(SideAngle)])

    return SideLengths


def GetMappingRatios(SideLengths, ActualSize):
    Ratios = []

    for Lengths in SideLengths:
        cosTheta = math.cos(math.radians(Lengths[-1]))
        Ratios.append([((ActualSize[0] * cosTheta) / Lengths[0]), ((ActualSize[1] * cosTheta) / Lengths[1])])

    return Ratios



def MapPixels(Image, ActualSize, NumOfAruco, DictionaryType = cv2.aruco.DICT_6X6_50):
    # FInding aruco markers
    FoundFlag, Corners, IDs = DetectAruco(Image.copy(), NumOfAruco, DictionaryType)
    if not FoundFlag:
        return False, None, None, None

    # Getting side length of the markers in pixels (horizontal, verticle)
    SideLengths = GetArucoSideLengthInPixels(Corners)

    # Getting Mapping ratios
    Ratios = GetMappingRatios(SideLengths, ActualSize)

    return True, Ratios, Corners, IDs


def MapPixels_Avg(Image, ActualSize, NumOfAruco, DictionaryType = cv2.aruco.DICT_6X6_50):
    Flag, Mapping, Corners, IDs = MapPixels(Image, ActualSize, NumOfAruco, DictionaryType=DictionaryType)
    if not Flag:
        return Flag, Mapping, Corners, IDs
    
    Mapping = np.mean(np.asarray(Mapping), axis=0)

    return Flag, Mapping, Corners, IDs



if __name__ == "__main__":
    Image = cv2.imread("Aruco_5x5_cm.png", cv2.IMREAD_UNCHANGED)

    Flag, Mapping, _, _ = MapPixels_Avg(Image, (5, 5), 5)
    if not Flag:
        print("\nAruco markers not found properly.")
        print("Retake the image.\n")

    print("Mapping (Horizontal, Verticle): {}".format(Mapping))
