import os
import cv2
import numpy as np
from scipy.stats import mode, norm
from PIL import Image
import matplotlib.pyplot as plt
from pytesseract import image_to_string, image_to_boxes

class Reg_Calculation(object):

    def __init__(self, mseReg, box):

        self.reg                        = mseReg
        self.a, self.b, self.c, self.d  = box
        self.place                      = self.Define_Area()
        self.circum                     = None

    def Define_Shape(self):

        a,b,c,d = cv2.boundingRect(self.reg)
        return a,b,c,d

    def Define_Area(self):

        return len(list(self.reg))

    def Return_Circum(self, Canny_Output):

        canny       = np.where(Canny_Output[self.b:self.b + self.d, self.a:self.a + self.c] != 0)[0]
        self.circum = len(canny)

        return self.circum

    def Return_Occup(self):

        temp = 1e-10 + self.c * self.d
        return self.place / temp

    def Return_Ratio(self):

        maxvalue = max(self.c, self.d)
        minvalue = min(self.c, self.d) + 1e-10
        result   = maxvalue / minvalue

        return  result

    def Return_Comp(self):

        if not self.circum:
            return None
        else:
            temp = 1e-10 + pow(self.circum, 2)
            return self.place / temp

    def Return_Hardness(self):

        temp = 1e-10 + self.c * self.d
        return self.place / temp

    def RGB(self, Input_Img):

        for i in range(0,3):
            Input_Img[self.reg[:, 1], self.reg[:, 0], i] = np.random.randint(low=120, high=256)

        return Input_Img



class TextDetection(object):

    def __init__(self, inputfile):

        Area_Limit                          = 2.0e-4
        Perimeter_Limit                     = 1e-4
        Ratio_Limit                         = 5
        Occupation_Range                    = (0.23, 0.9)
        Compact_Range                       = (3e-3, 1e-1)
        Swt_Count_Limit                     = 10
        Swt_Std_Limit                       = 20
        Stroke_Size_Limit                   = 0.02       
        Stroke_Variance_Limit               = 0.15     
        Step_Count                          = 10
        K                                   = 3
        Repeat_Time                         = 7
        Gain                                = 10

        self.inputfile                      = inputfile
        img                                 = cv2.imread(inputfile)
        color_img                           = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img                            = color_img
        self.d, self.c                      = img.shape[:2]

        self.Area_Limit                     = Area_Limit
        self.Perimeter_Limit                = Perimeter_Limit
        self.Ratio_Limit                    = Ratio_Limit
        self.Occupation_Range               = Occupation_Range
        self.Compact_Range                  = Compact_Range
        self.Swt_Count_Limit                = Swt_Count_Limit
        self.Swt_Std_Limit                  = Swt_Std_Limit
        self.Stroke_Size_Limit              = Stroke_Size_Limit
        self.Stroke_Variance_Limit          = Stroke_Variance_Limit
        self.Step_Count                     = Step_Count
        self.K                              = K
        self.Repeat_Time                    = Repeat_Time
        self.Gain                           = Gain

        self.Finalimg                       = color_img.copy()

        self.height, self.width             = self.img.shape[:2]

        self.grayscale                      = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)
        #--------------------------------------------------------------------------------------------------
        sigma                               = 0.36
        Bot_ratio                           = 1.0 - sigma
        Top_ratio                           = 1.0 + sigma
        Bottomtemp                          = Bot_ratio * np.median(self.img)
        Bottom                              = int(max(0, Bottomtemp))
        Toptemp                             = Top_ratio * np.median(self.img)
        Top                                 = int(min(255, Toptemp))
        self.Canny_Output                   = cv2.Canny(self.img, Bottom, Top)
        #--------------------------------------------------------------------------------------------------

        self.Sobel_X                        = cv2.Sobel(self.grayscale, cv2.CV_64F, 1, 0, ksize=-1)
        self.Sobel_Y                        = cv2.Sobel(self.grayscale, cv2.CV_64F, 0, 1, ksize=-1)

        self.sX                             = self.Sobel_Y.astype(int) 
        self.sY                             = self.Sobel_X.astype(int)

        Xtemp                               = self.sX * self.sX
        Ytemp                               = self.sY * self.sY
        self.Mag                            = np.sqrt( Xtemp + Ytemp )

        Magcalculation                      = (self.Mag + 1e-10)
        self.gX                             = self.sX / Magcalculation
        self.gY                             = self.sY / Magcalculation

    def Return_Stroke_Prop(self, sws):

        if len(sws) == 0:
            return (0, 0, 0, 0, 0, 0)

        try:
            mpsw        = mode(sws, axis=None)[0][0]
            mpswc       = mode(sws, axis=None)[1][0]

        except IndexError:
            mpsw        = 0
            mpswc       = 0

        try:
            swsmean, swsstd   = norm.fit(sws)
            swsmin       = int(min(sws))
            swsmax       = int(max(sws))

        except ValueError:
            swsmean        = 0
            swsstd         = 0
            swsmin       = 0
            swsmax       = 0

        return mpsw, mpswc, swsmean, swsstd, swsmin, swsmax

    def Return_Stroke(self, components):

        a, b, c, d  = components
        sws         = np.array([[np.Infinity, np.Infinity]])
        
        for i in range(b, b + d):
            for j in range(a, a + c):

                Cannytemp   = self.Canny_Output[i, j]

                if Cannytemp != 0:

                    g_X     = self.gX[i, j]
                    g_Y     = self.gY[i, j]

                    pX      = i
                    pY      = j
                    pXo     = i
                    pYo     = j
                    ss      = 0

                    go      = True 
                    goo     = True

                    sw      = np.Infinity
                    swo     = np.Infinity

                    while (go or goo) and (ss < self.Step_Count):

                        ss = ss + 1

                        if go:
                            cxtemp      = g_X * ss
                            cytemp      = g_Y * ss
                            cX          = np.int(np.floor(i + cxtemp))
                            cY          = np.int(np.floor(j + cytemp))
                            cxTF        = cX <= b or cX >= b + d
                            cyTF        = cY <= a or cY >= a + c

                            if ( cxTF or cyTF):
                                go      = False

                            cxyTF       = ((cX != pX) or (cY != pY))
                            if go and cxyTF:

                                try:
                                    cannyxy         = self.Canny_Output[cX, cY]

                                    if cannyxy      != 0:
                                        gx          = g_X * -self.gX[cX, cY]
                                        gy          = g_Y * -self.gY[cX, cY]
                                        gxy         = gx + gy
                                        pitemp      = np.pi / 2.0

                                        if np.arccos( gxy ) < pitemp:
                                            cxp     = pow((cX - i),2)
                                            cyp     = pow((cY - j),2)
                                            cxyp    = cxp + cyp
                                            sw      = int(np.sqrt( cxyp ))
                                            go      = False

                                except IndexError:
                                    go = False

                                pX = cX
                                pY = cY

                        if goo:
                            cxtemp       = g_X * ss
                            xytemp       = g_Y * ss                            
                            cXo          = np.int(np.floor(i - cxtemp))
                            cYo          = np.int(np.floor(j - cytemp))
                            cxoTF        = cXo <= b or cXo >= b + d
                            cyoTF        = cYo <= a or cYo >= a + c

                            if ( cxoTF or cyoTF):
                                goo      = False

                            cxyoTF       = ((cXo != pXo) or (cYo != pYo))
                            if goo and cxyoTF:

                                try:
                                    cannyxyo        = self.Canny_Output[cXo, cYo]

                                    if cannyxyo     != 0:
                                        gxo         = g_X * -self.gX[cXo, cYo]
                                        gyo         = g_Y * -self.gY[cXo, cYo]
                                        gxyo        = gxo + gyo
                                        pitemp      = np.pi / 2.0                    

                                        if np.arccos( gxyo ) < pitemp:
                                            cxop     = pow((cXo - i),2)
                                            cyop     = pow((cYo - j),2)
                                            cxyop    = cxop + cyop
                                            swo      = int(np.sqrt( cxyop ))
                                            goo = False

                                except IndexError:
                                    goo = False

                                pXo = cXo
                                pYo = cYo

                    sws = np.append(sws, [(sw, swo)], axis=0)

        swso = np.delete(sws[:, 1], np.where(sws[:, 1] == np.Infinity))
        sws = np.delete(sws[:, 0], np.where(sws[:, 0] == np.Infinity))

        return sws, swso

    def detect(self):

        restemp = np.zeros_like(self.img)

        mser             = cv2.MSER_create()
        regs, boxs       = mser.detectRegions(self.grayscale)

        mseregs          = len(regs)
        finalregs        = 0

        for i, (reg, box) in enumerate(zip(regs, boxs)):

            reg                     = Reg_Calculation(reg, box)

            cdatemp                 = self.c * self.d * self.Area_Limit
            AreaTF                  = reg.place < cdatemp

            PerimTF1                = reg.Return_Circum(self.Canny_Output)
            cdtemp                  = (self.c + self.d)
            PerimTF2                = (2 * cdtemp * self.Perimeter_Limit)
            PerimTF                 = PerimTF1 < PerimTF2

            RatioTF                 = reg.Return_Ratio() > self.Ratio_Limit

            OccupTF1                = (reg.Return_Occup() < self.Occupation_Range[0])
            OccupTF2                = (reg.Return_Occup() > self.Occupation_Range[1])
            OccupTF                 =  OccupTF1 or OccupTF2

            CompTF1                 = (reg.Return_Comp() < self.Compact_Range[0])
            CompTF2                 = (reg.Return_Comp() > self.Compact_Range[1])
            CompTF                  = CompTF1 or CompTF2

            if  AreaTF or PerimTF or RatioTF or OccupTF or CompTF:
                continue

            a, b, c, d              = box

            sws, swso               = self.Return_Stroke((a, b, c, d))

            sw, swc, _, swsstd, _, _= self.Return_Stroke_Prop(sws)

            swo, swco, _, so, _, _  = self.Return_Stroke_Prop(swso)

            if swco > swc:    
                sws                 = swso
                sw                  = swo
                swc                 = swco
                swsstd              = so

            TF1                     = len(sws) < self.Swt_Count_Limit
            TF2                     = swsstd > self.Swt_Std_Limit
            TF3temp                 = sw / max(reg.c, reg.d)
            TF3                     = TF3temp < self.Stroke_Size_Limit

            if TF1 and TF2 and TF3:
                continue
            
            VC = sw / (swsstd * swsstd + 1e-10)

            if  VC > self.Stroke_Variance_Limit:
                finalregs   =  finalregs + 1
                restemp     = reg.RGB(restemp)

        print("{} Final Total Regions.".format(finalregs))

        DoBinary                    = np.zeros_like(self.grayscale)
        r, c, _                     = np.where(restemp != [0, 0, 0])
        DoBinary[r, c]              = 255


        filter_kernel               = np.zeros((self.K, self.K), dtype=np.uint8)
        kenerltemp                  = self.K // 2
        filter_kernel[(kenerltemp)] = 1

        finalresult                 = np.zeros_like(self.grayscale)
        dialtetemp                  = cv2.dilate(DoBinary.copy(), filter_kernel, iterations= self.Repeat_Time)
        conts, hieras               = cv2.findContours(dialtetemp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i, (cont, hiera) in enumerate(zip(conts, hieras[0])):

            if hiera[-1] != -1:
                continue

            min_temp                = cv2.minAreaRect(cont)
            box                     = cv2.boxPoints(min_temp)
            box                     = np.int0(box)
            cv2.drawContours(self.Finalimg, [box], 0, (0, 255, 0), 2)
            cv2.drawContours(finalresult, [box], 0, 255, -1)

        return finalresult

def plt_show(*inputimages):

    imagelength     = len(inputimages)
    ilength         = imagelength / 3.
    R               = np.ceil(ilength)

    for i in range(imagelength):
        plt.subplot(R, 3, i + 1)

        if len(inputimages[i][0].shape) == 2:
            plt.imshow(inputimages[i][0], cmap='gray')

        else:
            plt.imshow(inputimages[i][0])

        plt.xticks([])
        plt.yticks([])

        plt.title(inputimages[i][1])

    plt.show()


inputfile       = "T4.jpg"

Structure       = TextDetection(inputfile)

finalresult     = Structure.detect()

plt_show((Structure.img, "Original"), (Structure.Finalimg, "Final Image"), (finalresult, "Mask For Transformation"))

#---------------------------------------------------------------------------------------------------------------

MSER_Temp       = cv2.MSER_create()
inputimg        = cv2.imread(inputfile)

grayimg         = cv2.cvtColor(inputimg, cv2.COLOR_BGR2GRAY)
OCR             = inputimg.copy()

regs, _         = MSER_Temp.detectRegions(grayimg)
HullContents    = [cv2.convexHull(rs.reshape(-1, 1, 2)) for rs in regs]

cv2.polylines(OCR, HullContents, 1, (0, 255, 0))
cv2.imshow('OCR image', OCR)
cv2.waitKey(0)

cover           = np.zeros((inputimg.shape[0], inputimg.shape[1], 1), dtype= np.uint8)

for HCs in HullContents:
    cv2.drawContours(cover, [HCs], -1, (255, 255, 255), -1)

FixedFinal      = cv2.bitwise_and(inputimg, inputimg, mask= cover)
cv2.imshow("Fixed Image", FixedFinal)
cv2.waitKey(0)