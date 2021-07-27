import cv2
import numpy as np
import face_recognition

from .IDetect import IDetect
from .core import Core
class FaceDetectFRLib(Core, IDetect):
    
    def __init__(self, source, method):
        super().__init__(source = source)
        self.face = []
        self.bboxes = []
        self.method = method
        
        if 0 == method:
            self.mod = "hog"
        elif 1 == method:
            self.mod = "cnn"
        
    #end

    def faceDetect(self, img = np.zeros([10,10])):
        imgC = img.copy()
        self.face = []
        self.bboxes = []
        fx = 1
        fy = 1

        frame = cv2.resize(imgC, (0, 0), fx=fx, fy=fy)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faceLoc = face_recognition.face_locations(frame, model = self.mod)

        for (top, right, bottom, left) in faceLoc:
            #Eğer görüntü boyutlandırılsa yüzün tam konumu göstermek için, ne kadar resmi küçülttüysek, ters oranda yüz konumlarını çarpıyoruz
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1
            
            #Yüzü çerçeve içerisine al
            imgC = cv2.rectangle(imgC, (left, top), (right, bottom), (0, 255, 255), 1,1)
           
            bbox = (left,top,right,bottom)
            self.bboxes.append(bbox)
                   
        self.face = np.array(imgC)
        return self.bboxes, self.face
    
    
    def run(self):
        while True:
            success, frame = self.vc.read()
            a,face = self.faceDetect(frame)
            if face == []:
                face = frame
                
            print(face)
            cv2.imshow("Ekran", face)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.vc.release()
    #end
    
#end::cl