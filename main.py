import cv2
import numpy as np
import argparse

import pixel_length_mapping as plm



def ArgParse():
        global args

        # Construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()

        ap.add_argument("imgPath", help="Path of the input image")
        ap.add_argument("-d", "--Diameter", type=int, help="Diameter used in cellpose")
        ap.add_argument("-rf", "--RescaleFactor", type=float, default=1, help="Rescale factor of the image")

        args = vars(ap.parse_args())



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



def SegmentCellpose(img, Diameter):
    import cellpose
    from cellpose import utils, core, models


    # Global variables shifted here
    use_GPU = core.use_gpu()
    print('GPU activated? %d'%use_GPU)

    # model_type='cyto' or model_type='nuclei'
    Model = models.Cellpose(gpu=use_GPU, model_type='cyto', net_avg=False)
    
    masks, flows, styles, diams = Model.eval(img, diameter=Diameter, channels=[0, 0], do_3D=False)
    outlines = cellpose.utils.outlines_list(masks)
    
    Contours = []
    for outline in outlines:
        contour = []
        for i in range(len(outline)):
            contour.append(np.array([np.asarray([outline[i][0], outline[i][1]], dtype=np.int32)], dtype=np.int32))
        Contours.append(np.asarray(contour, dtype=np.int32))

    return Contours


def main(imgPath, Diameter, Rescale_Factor):
    img = cv2.imread(imgPath)
    img = cv2.resize(img, (0, 0), fx=Rescale_Factor, fy=Rescale_Factor)

    Contours = SegmentCellpose(img, Diameter)

    # Map pixel length
    Flag, Mapping, Corners, _ = plm.MapPixels_Avg(img, (5, 5), 1)
    if not Flag:
        print("\nAruco markers not found properly.")
        print("Retake the image.\n")

    Contours = DeleteMarkerContour(Contours, Corners)
    
    finalData = getData(Contours, Mapping)

    return finalData


#print(main("GramsAppImages/mung_good_final_wia.jpg", 20, 0.25))
if __name__ == "__main__":
        ArgParse()

        results = main(args["imgPath"], args["Diameter"], args["RescaleFactor"])
        print(results)