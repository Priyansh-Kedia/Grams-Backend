try:
    import os
    import cv2
    import math
    import argparse
    import numpy as np
    from matplotlib import pyplot as p
except:
    raise ValueError("Libraries not installed.")

try:
    import ArucoBox
    import poseSegmentation
except:
    raise ValueError("Other files not found.")


def ArgParse():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("imgPath", help="Path of the input image")

    args = vars(ap.parse_args())

    return args


def Distance(Pt1, Pt2):
    return math.sqrt( ( (Pt1[1] - Pt2[1]) ** 2 ) + ( (Pt1[0] - Pt2[0]) ** 2 ) )

def TransformBox(Image, BoxCorners):
    # Assumed that orientation is clockwise and starting from top left
    Width = int((Distance(BoxCorners[0], BoxCorners[1]) + Distance(BoxCorners[2], BoxCorners[3])) / 2)
    Height = int((Distance(BoxCorners[1], BoxCorners[2]) + Distance(BoxCorners[3], BoxCorners[0])) / 2)
    if Width > Height:
        Width, Height = Height, Width
        BoxCorners = BoxCorners[1:] + BoxCorners[:1]

    FinalPoints = np.float32([[0, 0],
                              [Width-1, 0], 
                              [0, Height-1], 
                              [Width-1, Height-1]])

    # Correcting order of initial points (BoxCorners)
    BoxCorners[2], BoxCorners[3] = BoxCorners[3], BoxCorners[2]
    BoxCorners = np.asarray(BoxCorners, dtype=np.float32)

    # Applying prespective transformation.
    ProjectiveMatrix = cv2.getPerspectiveTransform(BoxCorners, FinalPoints)
    ImageTransformed = cv2.warpPerspective(Image, ProjectiveMatrix, (Width, Height))

    return ImageTransformed


def ExtractBox(Image, minBoxDimsPer_Ori=0.95, ThVal=100):
    if not (0 <= ThVal <= 255):
        return None
    Height, Width = Image.shape[:2]
    
    # Converting to gray
    GrayImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    while True:
        minBoxDimsPer = minBoxDimsPer_Ori

        # Thresholding for box
        Th = cv2.threshold(GrayImage, ThVal, 255, cv2.THRESH_BINARY_INV)[1]

        # Getting contours and finding the box's contour
        Contours = cv2.findContours(Th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        BBs = [cv2.boundingRect(Contour) for Contour in Contours]
        
        Box_Contour = None
        while minBoxDimsPer > 0.5 and Box_Contour is None:
            min_Box_Height, min_Box_Width = int(Height * minBoxDimsPer), int(Width * minBoxDimsPer)
            for i in range(len(BBs)):
                if BBs[i][2] >= min_Box_Width and BBs[i][3] >= min_Box_Height:
                    Box_Contour = Contours[i].copy()
                    break
            minBoxDimsPer -= 0.03

        if Box_Contour is not None:
            break
        ThVal += 10

    BB = cv2.boundingRect(Box_Contour)
    ImageCropped = Image[BB[1]+2 : BB[1]+BB[3]-2, BB[0]+2 : BB[0]+BB[2]-2].copy()

    return ImageCropped


def TransformExtractBox_GetMap(Image, OriDistBetArucoCorners=(23.47, 17.26)):
    # Getting aruco box points
    BoxCorners = ArucoBox.GetBox(Image, NumOfMarkers=4)
    if BoxCorners is None:
        raise ValueError("All markers not found properly. Retake the image.")

    # Transforming image
    Image = TransformBox(Image, BoxCorners.copy())

    # Getting Map : (Pixel Height, Pixel Width)
    h, w = Image.shape[:2]
    PixelMapping = (OriDistBetArucoCorners[0] / h, OriDistBetArucoCorners[1] / w)
    
    # # Extracting the box
    # Image = ExtractBox(Image)

    return Image, PixelMapping


def CalcRescaleFactors(ImgH, ImgW, RefH=1755, RefW=1275):
    RatioW = RefW / ImgW
    RatioH = RefH / ImgH

    Ratio = min(RatioW, RatioH)

    RF = float(str(0.05 * round(Ratio / 0.05))[:5])
    return RF, 1/RF


def NumberGrain(Image, Contours, StartingIndex=1, FontScale=0.65, Thickness=1):
    Font = cv2.FONT_HERSHEY_SIMPLEX
    Colour = (0, 0, 255)

    ImageCopy1 = Image.copy()
    cv2.drawContours(ImageCopy1, Contours, -1, (0, 255, 0), 1)

    for i in range(len(Contours)):
        Contour = Contours[i]
        x, y, w, h = cv2.boundingRect(Contour)
        BottomLeftPoint = ( x + w//5, y + h//2)
        ImageCopy1 = cv2.putText(ImageCopy1, str(i+StartingIndex), BottomLeftPoint, Font, FontScale, Colour, Thickness, cv2.LINE_AA)

    return ImageCopy1


def FindParams(Contour, Mapping):
    ParamsList = []

    # Area
    Area = cv2.contourArea(Contour)
    ParamsList.append(Area * Mapping[0] * Mapping[1])

    # Width and Height
    Rect = cv2.minAreaRect(Contour)
    ParamsList.append(Rect[1][0] * Mapping[1])       # Width
    ParamsList.append(Rect[1][1] * Mapping[0])       # Height
    try:
        ParamsList.append((Rect[1][0] * Mapping[1]) / (Rect[1][1] * Mapping[0]))       # Width/Height
    except:
        ParamsList.append(-1)

    # Circularity
    Perimeter = cv2.arcLength(Contour, True)
    try:
        Circularity = 4 * np.pi * (Area / (Perimeter * Perimeter))
    except:
        Circularity = 1.0
    ParamsList.append(Circularity)

    return ParamsList


def GetAvg(Contour_Dict):
    Values = np.asarray(list(Contour_Dict.values()))
    Mean = np.mean(Values, axis=0)

    return Mean


def getData(Contours, Mapping):
    # Storing data in csv
    Contour_Dict = dict()
    
    for i in range(len(Contours)):
        Contour_Dict[str(i)] = FindParams(Contours[i], Mapping)
    
    try:
        from pandas import DataFrame, Series
    except:
        raise ValueError("Pandas not found.")

    df = DataFrame(dict([ (k,Series(v)) for k,v in Contour_Dict.items() ]))
    df = df.T
    df.to_csv("Grain_AppData.csv", header=False, index=False) 
    
    Mean = GetAvg(Contour_Dict)
    
    return list([len(Contours)]) + list(Mean)


def main(ImagePath):
    # # Checking image path
    # if ImagePath is None or not os.path.exists(ImagePath):
    #     raise ValueError("Image path is not correct.")

    # Reading image
    Image = cv2.imread("testImage_Mung.jpg")
    # Image = cv2.imread(ImagePath)

    # Checking image
    if Image is None:
        raise ValueError("Image not read.")

    # Transform and extract box
    Image, PixelMapping = TransformExtractBox_GetMap(Image)

    # Calculating downcale and upscale factors
    DownscaleFactor, UpscaleFactor = CalcRescaleFactors(Image.shape[0], Image.shape[1])

    # Applying cellpose segmentation
    Outlines = poseSegmentation.ApplyCellpose(Image, DownscaleFactor=DownscaleFactor, UpscaleFactor=UpscaleFactor, 
                                              SaveOutlines=True, ShowContoursImage=False, CorrectOutlinesFlag=True)

    # Checking if any outline found
    if Outlines is None or len(Outlines) == 0:
        return [0.0] * 6, Image.copy()

    # Numbering grains
    NumberedImage = NumberGrain(Image, Outlines)

    # Getting grain data
    GrainData = getData(Outlines, PixelMapping)

    return GrainData, NumberedImage


if __name__ == "__main__":
    args = ArgParse()

    results, _ = main(args["imgPath"])
