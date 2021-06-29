import cv2
import numpy as np

import njitFunc


def MaskImage_Gray_Otsu(Image):
    # Getting mask image
    Gray_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    MaskImage = cv2.threshold(Gray_Image, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    # kernel = np.ones((3, 3), dtype=np.uint8)
    # MaskImage = cv2.erode(MaskImage, kernel, iterations=3)
    # MaskImage = cv2.dilate(MaskImage, kernel, iterations=2)

    return MaskImage


def MaskImage_HSV_Otsu(Image):
    # Getting mask image
    HSV_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)
    th0 = cv2.threshold(HSV_Image[:, :, [0]], 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    th1 = cv2.threshold(HSV_Image[:, :, [1]], 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    MaskImage = cv2.bitwise_or(th0, th1)

    kernel = np.ones((3, 3), dtype=np.uint8)
    # MaskImage = cv2.erode(MaskImage, kernel, iterations=1)
    # MaskImage = cv2.erode(MaskImage, kernel, iterations=3)
    # MaskImage = cv2.dilate(MaskImage, kernel, iterations=2)

    return MaskImage


#########################################################################################


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

#####################################################################################################################

priority_inner = np.array([[ 6,  7, 10, 12, 10,  7,  6],
                           [ 7, 13, 16, 17, 16, 13,  7],
                           [10, 16, 18, 19, 18, 16, 10],
                           [12, 17, 19, 20, 19, 17, 12],
                           [10, 16, 18, 19, 18, 16, 10],
                           [ 7, 13, 16, 17, 16, 13,  7],
                           [ 6,  7, 10, 12, 10,  7,  6]], dtype=np.uint8)

priority_outer = np.array([[ 5,  8,  9,  8,  5],
                           [ 8, 14, 15, 14,  8],
                           [ 9, 15, 20, 15,  9],
                           [ 8, 14, 15, 14,  8],
                           [ 5,  8,  9,  8,  5]], dtype=np.uint8)



def RemoveUnupdated(outline, ptPriority, n):
    i = 0
    new_outline = []

    while i < outline.shape[0]:
        # If updated, append outline point and continue
        if ptPriority[i] != 0:
            new_outline.append(outline[i])
            i += 1
            continue
        
        j = i
        count = 0
        while j < len(outline):
            if ptPriority[j] == 0:
                count += 1
                j += 1
            else:
                break
        if count > n:
            while i < j:
                new_outline.append(outline[i])
                i += 1
        else:
            i = j

    return np.array(new_outline)


def UpdateByMask(MaskImage, outline):
    # Getting mask with by outlines
    MaskByOutline = np.zeros(MaskImage.shape, dtype=np.uint8)
    cv2.drawContours(MaskByOutline, [outline], -1, 255, -1)

    # Performing bitwise and operation
    FinalMask = cv2.bitwise_and(MaskByOutline, MaskImage)

    # Getting the contours
    Contours = cv2.findContours(FinalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

    # Biggest Contour
    Contour = sorted(Contours, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    return Contour


def updateOutline(EdgeImage, outline, GrainMaskByOutline, GrainMaskImage, n=7, UpdateOutlineMask=True):
    EdgeImage //= 255
    ptPriority = np.zeros(outline.shape[0], dtype=np.uint8)

    # First pass - inside ones
    EdgeImage_inner = cv2.bitwise_and(EdgeImage, GrainMaskByOutline)
    outline, ptPriority = njitFunc.outlineCorrectionPass(EdgeImage_inner, outline, ptPriority, priority_inner)

    # Second Pass - outside ones
    EdgeImage_outer = cv2.bitwise_and(EdgeImage, cv2.bitwise_not(GrainMaskByOutline))
    outline, ptPriority = njitFunc.outlineCorrectionPass(EdgeImage_outer, outline, ptPriority, priority_outer)

    # Removing outline points that are not updated (point priority = 0)
    # If more than 'n' consecutive points are not updated, the they are considered as outline
    outline = RemoveUnupdated(outline, ptPriority, n)
    
    # Updating outline according to the mask image of the original input image
    if UpdateOutlineMask:
        outline = UpdateByMask(GrainMaskImage, outline)

    return outline


def CorrectOutlines(Image, outlines, UpdateOutlineMask=True):
    # Getting edge image
    EdgeImage = cv2.Canny(Image, 30, 70)
    EdgeImage = cv2.copyMakeBorder(EdgeImage, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)

    MaskImage = MaskImage_HSV_Otsu(Image)
    MaskImage = cv2.copyMakeBorder(MaskImage, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)

    # # drawing and saving edges and current outlines
    # EdgeImage2 = cv2.cvtColor(EdgeImage, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(EdgeImage2, outlines, -1, (0, 255, 0), 1)
    # cv2.imwrite("EdgeWithOutlines.jpg", EdgeImage2)


    new_outlines = []
    for outline in outlines:
        shiftedOutline, GrainMaskByOutline, BB = ShiftOutline(outline, AddBorder=True, BorderSize=5)

        GrainEdgeImage = EdgeImage[BB[1]:BB[1]+BB[3], BB[0]:BB[0]+BB[2]].copy()
        GrainMaskImage = MaskImage[BB[1]:BB[1]+BB[3], BB[0]:BB[0]+BB[2]].copy()
        updatedOutline = updateOutline(GrainEdgeImage, shiftedOutline, GrainMaskByOutline, GrainMaskImage, UpdateOutlineMask=UpdateOutlineMask)

        new_outline = RRevertOutlineShift(updatedOutline, BB)

        new_outlines.append(new_outline)

    return new_outlines


#########################################################################################################################


def ShiftOutline(Contour, ContourBoundarySize=-1, AddBorder=True, BorderSize=1):
    ShiftedContour = []								# Shifted contour will be stored here.
    (x1, y1, w, h) = cv2.boundingRect(Contour)		# Getting the bounding rectangle of the contour.

    if AddBorder:
        x1 -= BorderSize
        y1 -= BorderSize
        w += 2*BorderSize
        h += 2*BorderSize

    # Subtract top left corner of the bounding box to each point in the contour to get the shifted cotour.
    for i in range(len(Contour)):
        ShiftedContour.append(np.array([(Contour[i, 0] - x1), (Contour[i, 1] - y1)]))
    ShiftedContour = np.array(ShiftedContour)

    # Creating a black and white image with contour coloured completely as white and background as black.
    BlackImage = np.zeros((h, w, 1), dtype=np.uint8)
    cv2.drawContours(BlackImage, [ShiftedContour], -1, 255, ContourBoundarySize)

    # Returning data as required.
    return ShiftedContour, BlackImage, (x1, y1, w, h)



def RRevertOutlineShift(Contour, BoundingRect):
    (x1, y1, w, h) = BoundingRect

    RevertedShiftContours = []
    for i in range(len(Contour)):
        try:
            RevertedShiftContours.append([(Contour[i, 0, 0] + x1-1), (Contour[i, 0, 1] + y1)])
        except:
            RevertedShiftContours.append([(Contour[i, 0] + x1-1), (Contour[i, 1] + y1)])

    return np.array(RevertedShiftContours)


################################

def GetBackgroundColor(Image):
    B_Image = Image[:, :, 0].copy().astype(np.float32) / 255
    Height, Width = B_Image.shape[:2]

    if Height < 20 and Width < 20:
        print("\nTake big image.")
        exit()

    TB = [0, 10, 15, Height - 5, Height - 10, Height - 15]
    LR = [0, 10, 15, Width - 5, Width - 10, Width - 15]

    Color = 0
    Count = 0

    for y in TB:
        Color += sum(B_Image[y, :])
        Count += len(B_Image[y, :])

    for x in LR:
        Color += sum(B_Image[:, x])
        Count += len(B_Image[:, x])

    Avg_Color = int((Color / Count) * 255)

    if Avg_Color <= 100:
        return "BLACK"
    else:
        return "BLUE"


def GetMaskImage(Image):
    BGC = GetBackgroundColor(Image)

    if BGC == "BLACK":
        MaskImage = MaskImage_Gray_Otsu(Image)
    elif BGC == "BLUE":
        MaskImage = MaskImage_HSV_Otsu(Image)
    else:
        print("\nBackground colour not found properly.")
        print("Found Color: {}\n".format(BGC))
        exit()

    return MaskImage  


def CorrectOutlines_DirectlyWithMask(Image, RescaleFactor,  outlines):
    if RescaleFactor != 1:
        Image = cv2.resize(Image, (0, 0), fx=RescaleFactor, fy=RescaleFactor, interpolation=cv2.INTER_CUBIC)

    MaskImage = GetMaskImage(Image)

    for i in range(len(outlines)):
        _, OutlineMask, BB = ShiftOutline(outlines[i], AddBorder=False)
        '''print()
        print(BB)
        print(OutlineMask.shape)
        print(MaskImage[BB[1]:BB[1]+BB[3], BB[0]:BB[0]+BB[2]].shape)
        print()'''
        NewMask = cv2.bitwise_and(OutlineMask, MaskImage[BB[1]:BB[1]+BB[3], BB[0]:BB[0]+BB[2]])
        Contours = cv2.findContours(NewMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        if len(Contours) == 0:
            continue
        outlines[i] = RRevertOutlineShift(sorted(Contours, key=lambda x: cv2.contourArea(x))[-1], BB)

    return outlines, MaskImage

###################################
def PyInit_process_outlines():
    pass
