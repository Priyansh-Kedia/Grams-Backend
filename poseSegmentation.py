import cv2
import argparse
import time as t
import numpy as np

import process_outlines as po 

# Importing cellpose and installing dependencies if required
try:
    import Compiled as cc
except:
    print("\nInstall cellpose dependencies first.")
    exit()


# Parsing the arguments
def ArgParse():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("ImagePath", 
                    help="Path of the image file.")
    ap.add_argument("-df", "--DownscaleFactor", default=0.25,
                    help="Downscale factor value (Downscale image by). (Default: 0.25)")
    ap.add_argument("-uf", "--UpscaleFactor", default=4,
                    help="Upscale factor value (Upscale cellpose output by). (Default: 4)")
    ap.add_argument("-co", "--CorrectOutlines", action='store_true',
                    help="Correct outlines found. (Default: False)")
    ap.add_argument("-so", "--SaveOutlines", action='store_true',
                    help="Save the contours outlines in the CSV file. (Default: False)")
    ap.add_argument("-show", "--ShowContoursImage", action='store_true',
                    help="Show the detected contours drawn on an image. (Default: False)")
    ap.add_argument("-uom", "--UpdateOutlineMask", action='store_true',
                    help="Update the outlines using the mask image. (Default: False)")
                    

    args = vars(ap.parse_args())            # Converting it to dictionary.
    
    return args




def SaveOutlinesInCSV(Outlines, CSV_name="OutlinesFound.csv"):
    from pandas import DataFrame, Series
    Outline_Dict = dict()
    for i in range(len(Outlines)):
        Outline_Dict[str(i)] = Outlines[i].reshape( len(Outlines[i]) * 2 )

    df = DataFrame(dict([ (k,Series(v)) for k,v in Outline_Dict.items() ]))
    df = df.T
    df.to_csv(CSV_name, header=False, index=False) 




def ApplyCellpose(Image, DownscaleFactor=0.25, UpscaleFactor=4, SaveOutlines=True, 
                  ShowContoursImage=True, CorrectOutlinesFlag=True, UpdateOutlineMask=True):
    CellposeStart = t.time()
    # Rescaling image
    smallImage = cv2.resize(Image.copy(), (0, 0), fx=DownscaleFactor, fy=DownscaleFactor, interpolation=cv2.INTER_CUBIC)

    # Global variables shifted here
    use_GPU = cc.use_gpu()
    print('GPU activated? %d'%use_GPU)

    # model_type='cyto' or model_type='nuclei'
    Model = cc.Pose(gpu=use_GPU, model_type='cyto', net_avg=False, Upscale_Factor=UpscaleFactor)

    masks, flows, styles, diams = Model.eval(smallImage, diameter=13, channels=[0, 0], do_3D=False)
    '''if CorrectOutlinesFlag:
        masks = cv2.copyMakeBorder(masks, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)'''

    outlinesStart = t.time()
    outlines = po.outlines_list_Fast(masks, DownscaleFactor, UpscaleFactor)
    print("Time Taken for calculating outlines: {}".format(t.time() - outlinesStart))

    # Correcting outlines
    if CorrectOutlinesFlag:
        CorrectOutlinesStart = t.time()
        new_outlines, maskImage_HSV_Otsu = po.CorrectOutlines_DirectlyWithMask(Image, DownscaleFactor*UpscaleFactor, outlines.copy())
        # outlines = po.CorrectOutlines(Image, outlines, UpdateOutlineMask=UpdateOutlineMask)
        print("Correcting Outlines Time: {}".format(t.time() - CorrectOutlinesStart))

        '''# Shifting outlines back by 5 pixel (border was added)
        for i in range(len(outlines)):
            for j in range(len(outlines[i])):
                outlines[i][j] = [outlines[i][j][0] - 5, outlines[i][j][1] - 5]'''

    print("Total Cellpose Runtime: {}".format(t.time() - CellposeStart))

    if SaveOutlines:
        SaveOutlinesInCSV(outlines)

    if ShowContoursImage:
        im1 = Image.copy()
        RF = DownscaleFactor*UpscaleFactor
        if RF != 1:
            im1 = cv2.resize(im1, (0, 0), fx=RF, fy=RF, interpolation=cv2.INTER_CUBIC)
        im2 = im1.copy()
        cv2.drawContours(im1, outlines, -1, (0, 0, 255), 1)
        cv2.drawContours(im1, new_outlines, -1, (0, 255, 0), 1)
        from matplotlib import pyplot as p
        p.subplot(121)
        p.imshow(im1[:, :, ::-1])
        p.subplot(122)
        b, g, r = cv2.split(im2)
        im2 = cv2.merge((b, g, maskImage_HSV_Otsu))
        p.imshow(im2[:, :, ::-1])
        p.show()

    return outlines


if __name__ == "__main__":
    # Parsing command line arguments
    args = ArgParse()

    # Reading Image
    Image = cv2.imread(args["ImagePath"])

    ApplyCellpose(Image.copy(), args["DownscaleFactor"], args["UpscaleFactor"], SaveOutlines=args["SaveOutlines"], 
                  ShowContoursImage=args["ShowContoursImage"], CorrectOutlinesFlag=args["CorrectOutlines"], 
                  UpdateOutlineMask=args["UpdateOutlineMask"])