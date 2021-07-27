import torch
import cv2
import numpy as np
import time

tic = time.time()
# %% FACE DETECTION
from faceLib.faceDetectCV2 import FaceDetectCV2
from faceLib.faceDetectMediapipe import FaceDetectMediapipe
from faceLib.faceDetectFRLib import FaceDetectFRLib
from faceLib.faceDetectMTCNN import FaceDetectMTCNN
from faceLib.faceDetectDlib import FaceDetectDlib


# %% FACE RECOGNITION
from faceLib.faceRecognizeDlib import FaceRecognizeDlib
from faceLib.faceRecognizeOnnxVGG import FaceRecognizeOnnxVGG


# %% FACTORY
class MethodFactory(object):
    def __init__(self):
        self.detectNames = ['FaceDetectDlib', 'FaceDetectMTCNN', 'FaceDetectFRLib', 'FaceDetectMediapipe', 'FaceDetectCV2']
        self.recogNames = ['FaceRecognizeOnnxVGG', 'FaceRecognizeDlib']

    def create(self, r, d):
        try:
            print("#Method= "+d + " <-> " + r)
            # detect parametreleri
            det = globals()[d](source=0, method=1)
            # Recognize parametreleri
            newClass = globals()[r](det, 30, "../helper/trainImages")
            name = (str(d) + " <-> " + str(r))
            return newClass, name
        except:
            print("HATA/ #Method= "+r + " <-> " + d)
                   

    def createMatchedClass(self, detectNames=None, recogNames=None):
        combineClass = []
        combineClassName = []

        import time
        tic = time.time()
        if None == detectNames:
            detectNames = self.detectNames
        if None == recogNames:
            recogNames = self.recogNames

        for r in recogNames:
            for i,d in enumerate(detectNames):
                
                newClass, name = self.create(r, d)
                combineClass.append(newClass)
                combineClassName.append(name)
                
            if(0 == i):
                while True:
                    if 1 == newClass.basarili:
                        break
                    # end
                # end
            #end
                    
        toc = time.time()
        return combineClass, combineClassName, (toc-tic)
        
