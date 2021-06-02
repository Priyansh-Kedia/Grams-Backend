import cv2
import time as t
import numpy as np
import argparse

import pixel_length_mapping as plm



def ArgParse():
        # Construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()

        ap.add_argument("imgPath", help="Path of the input image")
        ap.add_argument("-d", "--Diameter", type=int, default=20, help="Diameter used in cellpose")
        ap.add_argument("-df", "--Downscale_Factor", type=float, default=0.25, help="Downscale factor of the image")
        ap.add_argument("-uf", "--Upscale_Factor", type=float, default=4, help="Upscale factor of the image")

        args = vars(ap.parse_args())

        return args


def GetAvg(Contour_Dict):
    Values = np.asarray(list(Contour_Dict.values()))

    Mean = np.mean(Values, axis=0)

    return Mean



def FindParams(Contour, Mapping):
    ParamsList = []

    # Area
    Area = cv2.contourArea(Contour)
    ParamsList.append(Area * Mapping[0] * Mapping[1])

    # Width and Length
    Rect = cv2.minAreaRect(Contour)
    ParamsList.append(Rect[1][0] * Mapping[0])       # Width
    ParamsList.append(Rect[1][1] * Mapping[1])       # Length
    ParamsList.append((Rect[1][0] * Mapping[0]) / (Rect[1][1] * Mapping[1]))       # Width/Length

    # Circularity
    Perimeter = cv2.arcLength(Contour, True)
    Circularity = 4 * np.pi * (Area / (Perimeter * Perimeter))
    ParamsList.append(Circularity)

    return ParamsList


def getData(Contours, Mapping):
    # Storing data in csv
    Contour_Dict = dict()
    
    for i in range(len(Contours)):
        Contour_Dict[str(i)] = FindParams(Contours[i], Mapping)
    
    from pandas import DataFrame, Series
    df = DataFrame(dict([ (k,Series(v)) for k,v in Contour_Dict.items() ]))
    df = df.T
    df.to_csv("Output.csv", header=False, index=False) 
    
    Mean = GetAvg(Contour_Dict)
    
    return list([len(Contours)]) + list(Mean)


def DeleteMarkerContour(Contours, Corners):
    NewContours = []

    for Contour in Contours:
        Flag = True      # Append if true
        for Corner in Corners:
            Coordinates = Corner[0][0]
            dist = cv2.pointPolygonTest(Contour, tuple(Coordinates), False)
            if dist != -1:
                Flag = False 
                break
        
        if Flag:
            NewContours.append(Contour)

    return np.asarray(NewContours, dtype=type(Contours))


def NumberGrain(Image, Upscale_Factor, Contours, StartingIndex=1):
    # Upscaling it again
    Image = cv2.resize(Image, (0, 0), fx=Upscale_Factor, fy=Upscale_Factor)

    Font = cv2.FONT_HERSHEY_SIMPLEX
    FontScale = 0.6
    Thickness = 1
    Colour = (0, 0, 255)

    ImageCopy1 = Image.copy()
    cv2.drawContours(ImageCopy1, Contours, -1, (0, 255, 0), 1)

    for i in range(len(Contours)):
        Contour = Contours[i]
        x, y, w, h = cv2.boundingRect(Contour)
        BottomLeftPoint = ( x + w//3, y + (2*h)//3)
        ImageCopy1 = cv2.putText(ImageCopy1, str(i+StartingIndex), BottomLeftPoint, Font, FontScale, Colour, Thickness, cv2.LINE_AA)

    cv2.imwrite("NumberedImg.jpg", ImageCopy1)

    return ImageCopy1



def outlines_list_Fast(masks, DF, UF):
    GrainDim = int(200 * DF * UF)

    uni = np.unique(masks, return_index=True)

    toXY = []
    for i in range(len(uni[1])):
        toXY.append([uni[1][i] % masks.shape[1], uni[1][i] // masks.shape[1]])

    outlines = []
    for i in range(1, len(uni[0])):
        startPt = toXY[i]
        y1 = max(startPt[1] - 5, 0)
        x1 = max(startPt[0] - (GrainDim * 2), 0)
        y2 = min(masks.shape[0]-1, startPt[1] + GrainDim*3)
        x2 = min(masks.shape[1]-1, startPt[0] + (GrainDim * 2))

        GrainROI = masks[y1:y2+1, x1:x2+1]

        GrainROI_n = GrainROI == uni[0][i]

        Contour = cv2.findContours(GrainROI_n.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2][0]

        outline = []
        for j in range(len(Contour)):
            outline.append(np.array([Contour[j][0][0] + x1, Contour[j][0][1] + y1], dtype=int))
        outlines.append(np.asarray(outline, dtype=int))

    return outlines



def SegmentCellpose(img, Diameter, Downscale_Factor, Upscale_Factor):
    import Compiled as cc

    # Global variables shifted here
    use_GPU = cc.use_gpu()
    print('GPU activated? %d'%use_GPU)

    # model_type='cyto' or model_type='nuclei'
    Model = cc.Pose(gpu=use_GPU, model_type='cyto', net_avg=False, Upscale_Factor=Upscale_Factor)

    masks, flows, styles, diams = Model.eval(img, diameter=Diameter, channels=[0, 0], do_3D=False)

    outlinesStart = t.time()
    outlines = outlines_list_Fast(masks, Downscale_Factor, Upscale_Factor)
    print("Time Taken for calculating outlines: {}".format(t.time() - outlinesStart))

    # Numbering grains
    NumberedImg = NumberGrain(img.copy(), Upscale_Factor, outlines, StartingIndex=1)

    Contours = []
    for outline in outlines:
        contour = []
        for i in range(len(outline)):
            contour.append(np.array([np.asarray([outline[i][0], outline[i][1]], dtype=np.int32)], dtype=np.int32))
        Contours.append(np.asarray(contour, dtype=np.int32))

    return Contours, NumberedImg


def main(ImagePath, Diameter, Downscale_Factor, Upscale_Factor=4):
    Image = cv2.imread(ImagePath)
    DownsizedImage = cv2.resize(Image, (0, 0), fx=Downscale_Factor, fy=Downscale_Factor)
    if (Downscale_Factor*Upscale_Factor) != 1.0:
        FinalImage = cv2.resize(Image, (0, 0), fx=(Downscale_Factor*Upscale_Factor), fy=(Downscale_Factor*Upscale_Factor))
    else:
        FinalImage = Image.copy()

    Contours, NumberedImage = SegmentCellpose(DownsizedImage, Diameter, Downscale_Factor, Upscale_Factor)

    # Map pixel length
    Flag, Mapping, Corners, _ = plm.MapPixels_Avg(FinalImage, (5, 5), 1)
    if not Flag:
        print("\nAruco markers not found properly.")
        print("Retake the image.\n")

    else:
        Contours = DeleteMarkerContour(Contours, Corners)
    
    finalData = getData(Contours, Mapping)

    return finalData, NumberedImage


if __name__ == "__main__":
        args = ArgParse()

        results, _ = main(args["imgPath"], args["Diameter"], args["Downscale_Factor"], Upscale_Factor=args["Upscale_Factor"])
        print(results)
