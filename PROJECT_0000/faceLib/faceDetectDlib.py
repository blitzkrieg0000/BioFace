import os
#os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
import dlib
import cv2
import numpy as np

from .IDetect import IDetect
from .core import Core
class FaceDetectDlib(Core, IDetect):
    
    def __init__(self, source, method):
        super().__init__(source = source)
        self.face = []
        self.bboxes = []
        self.method = method

        if 0 == method:
            self.detector = dlib.get_frontal_face_detector()
        elif 1 == method:
            self.detector = dlib.cnn_face_detection_model_v1(r"../helper/dat/mmod_human_face_detector.dat")
        
    #end

    def faceDetect(self, image = np.zeros([10,10])):
        imgC = []
        imgC = image.copy() #Gelen resmin referans alınmaması için
        gray = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
        
        self.face = []
        self.bboxes = []
        faceLoc = self.detector(gray,1)
        for rect in faceLoc:
            if self.method == 0:
                left=rect.left()
                top=rect.top()
                right=rect.right()
                bottom=rect.bottom()
            else:
                left=rect.rect.left()
                top=rect.rect.top()
                right=rect.rect.right()
                bottom=rect.rect.bottom()

            cv2.rectangle(imgC, (left, top), (right, bottom), (255, 0, 255), 2)
        
            bbox = (left,top,right,bottom)
            self.bboxes.append(bbox)
        
        self.face = np.array(imgC)
        return self.bboxes, self.face
    #end
    
    def run(self):
        while True:
            success, frame = self.vc.read()
            _, face = self.faceDetect(frame)
            if face == []:
                face = frame
                
            print(face)
            cv2.imshow("Ekran", face)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.vc.release()
    #end
    
#end::class