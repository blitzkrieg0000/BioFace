from warnings import catch_warnings
import cv2
import numpy as np
import os
os.add_dll_directory("C:/Users/BLITZKRIEG/anaconda3/envs/GPU/lib/site-packages/torch/lib")

from facenet_pytorch import MTCNN
from PIL import Image

from .IDetect import IDetect
from .core import Core
class FaceDetectMTCNN(Core, IDetect):

    def __init__(self, source, method):
        super().__init__(source=source)
        self.face = []
        self.bboxes = []
        self.method = method
        self.detector = []
        try:
            self.detector = MTCNN(margin=20, keep_all=True, post_process=True, device='cuda')
        except:
            pass
    # end

    def faceDetect(self, img= np.zeros([10,10])):
        imgC = img.copy()
        
        self.bboxes=[]
        imgRGB = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(imgRGB)
        faceLoc, probs, landmarks = self.detector.detect(frame, landmarks=True)
        
        try:
            for i, box in enumerate(faceLoc):
                (x1, y1, w, h) = box
                (x1, y1, w, h) = int(abs(x1)), int(abs(y1)), int(abs(w)), int(abs(h))
                bbox = (x1, y1, w, h)
                self.bboxes.append(bbox)
                
                # Yüzü çerçeve içerisine al
                imgC = cv2.rectangle(imgC, (x1, y1), (w, h), (0, 200, 255), 2)
        except:
            print("Hata: faceDetect")
            
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
    # end

# end::cl