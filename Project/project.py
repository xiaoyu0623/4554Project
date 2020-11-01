import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image

class Library:

    Area_Limit = 2.0e-4
    Perimeter_Limit = 1e-4
    Ratio_Limit = 5.0
    Occupation_Range = (0.23, 0.90)
    Compact_Range = (3e-3, 1e-1)

class RegionCalculation(object):

    def __init__(self, Region, Box):
        self.reg = Region
        self.x, self.y, self.w, self.h = Box
        self.Area = self.Define_Area()
        self.Perimeter = None

    def Define_Area(self):
        return len(list(self.reg))

    def Return_Perim(self, canny_img):
        self.Perimeter = len(np.where(canny_img[self.y:self.y + self.h, self.x:self.x + self.w] != 0)[0])
        return self.Perimeter

    def Return_Occup(self):
        return self.Area / (self.w * self.h + 1e-10)

    def Return_Ratio(self):
        return max(self.w, self.h) / (min(self.w, self.h) + 1e-10)

    def Return_Comp(self):
        if not self.Perimeter:
            return None
        else:
            return self.Area / (self.Perimeter ** 2 + 1e-10)


def To_Canny(img):
    sigma = 0.36
    Bot_ratio = 1.0 - sigma
    Top_ratio = 1.0 + sigma
    Bottom = int(max(0, Bot_ratio * np.median(img)))
    Top = int(min(255, Top_ratio * np.median(img)))
    return cv2.Canny(img, Bottom, Top)


def Detection(inputfile, library):

        img = cv2.imread(inputfile)
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = color_img
        h, w = img.shape[:2]
        grayscale = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)

        library = library
        Area_Limit = library.Area_Limit
        Perimeter_Limit = library.Perimeter_Limit
        Ratio_Limit = library.Ratio_Limit
        Occupation_Range = library.Occupation_Range
        Compact_Range = library.Compact_Range

        grayscale = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        cannyscale = To_Canny(img)
        #cv2.imwrite("aftercanny.jpg", cannyscale)

        sobelX = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=-1)
        sobelY = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=-1)

        ###########################################################################################

        MSER_temp = cv2.MSER_create()
        Reg, Box = MSER_temp.detectRegions(grayscale)

        for k, (reg, box) in enumerate(zip(Reg, Box)):

            reg = RegionCalculation(reg, box)

            AreaTF = w * h * Area_Limit > reg.Area

            PerimTF = (2 * (w + h) * Perimeter_Limit) > reg.Return_Perim(cannyscale)

            RatioTF = Ratio_Limit < reg.Return_Ratio()

            OccupTF = (reg.Return_Occup() < Occupation_Range[0]) or (reg.Return_Occup() > Occupation_Range[1])

            CompTF = (reg.Return_Comp() < Compact_Range[0]) or (reg.Return_Comp() > Compact_Range[1])

            if  AreaTF or PerimTF or RatioTF or OccupTF or CompTF:
                continue
            
            a, b, c, d = box


        ###########################################################################################



###########################################################################################

inputimage = "Test1.jpg"
Lib = Library()
result = Detection(inputimage, Lib)

