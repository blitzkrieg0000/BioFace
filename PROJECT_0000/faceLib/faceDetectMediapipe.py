import cv2
import numpy as np
import mediapipe as mp

from .IDetect import IDetect
from .core import Core
class FaceDetectMediapipe(Core, IDetect):
    
    def __init__(self, source, method):
        super().__init__(source = source)
        self.face = []
        self.bboxes = []
        self.method = method
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.55)
        self.mpDraw = mp.solutions.drawing_utils
        
        if 0 == method:
            pass
        elif 1 == method:
            pass
    #end
    
    def faceDetect(self, img):
        self.face = []
        self.bboxes = []
        
        ih, iw, ic = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
        try:
            for id, detection in enumerate(results.detections):       
                #self.mpDraw.draw_detection(img, detection)
                faceLoc = detection.location_data.relative_bounding_box
                
                bbox = [int(faceLoc.xmin * iw), int(faceLoc.ymin * ih), int(faceLoc.width * iw), int(faceLoc.height * ih)]   
                (left,top,right,bottom) = (bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1]+bbox[2])
                
                cv2.rectangle(img, (left,top), (right ,bottom), (255, 255, 255), 2)  
                bbox = (left,top,right,bottom)
                self.bboxes.append(bbox)    
                
            self.face = np.array(img)
        except:
            print("mediapipeDetectFunc")
            self.face = np.array(img)
            
        return self.bboxes, self.face
    
    
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
    
#end::cl