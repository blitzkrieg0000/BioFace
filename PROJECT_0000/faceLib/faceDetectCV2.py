import os
#os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))
import dlib
import cv2
import numpy as np

from .IDetect import IDetect
from .core import Core
class FaceDetectCV2(Core, IDetect):
    
    def __init__(self, source, method):
        super().__init__(source = source)
        self.face = []
        self.bboxes = []
        self.method = method
        self.pose_predictor = dlib.shape_predictor(r"../helper/dat/shape_predictor_68_face_landmarks.dat")
        
        self.modelFile = "../helper/dat/res10_300x300_ssd_iter_140000.caffemodel"
        self.configFile = "../helper/dat/res10_300x300_ssd_iter_140000.prototxt"
        
        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            print("GPU Ayarlanamıyor!")
        
    #end

    def faceDetect(self, image = np.zeros([10,10])):
        imgC = image.copy() #Gelen resmin, referans alınmaması için
    
        self.face = []
        self.bboxes = []
        
        h, w = imgC.shape[:2]; 
        blob = cv2.dnn.blobFromImage(cv2.resize(imgC, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        
        try:
            for i in range(faces.shape[2]):
                    confidence = faces[0, 0, i, 2]

                    if confidence > 0.73:
                        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (left, top, right, bottom) = box.astype("int")
                        imgC = cv2.rectangle(imgC, (left, top), (right, bottom), (50, 200, 200), 2)
                        bbox = (left,top,right,bottom)
                        self.bboxes.append(bbox)

        except:
            pass
        
        self.face = np.array(imgC)
        return self.bboxes, self.face
    #end
    
    def run(self):
        while True:
            success, frame = self.vc.read()
            a, face = self.faceDetect(frame)
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