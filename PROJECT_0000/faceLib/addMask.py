from facenet_pytorch import MTCNN
import cv2
import numpy as np
import os
import dlib
import csv

os.add_dll_directory("C:/Users/BLITZKRIEG/anaconda3/envs/GPU/lib/site-packages/torch/lib")

class AddMask(object):
    
    def __init__(self, maskPath = "./"):
        self.maskPath = maskPath
        self.streamURL = 0
        self.detector = MTCNN(margin=20, keep_all=True, post_process=True, device='cuda')
        self.pose_predictor = dlib.shape_predictor("../helper/dat/shape_predictor_68_face_landmarks.dat")
        

        (self.srcMasks, self.pointMasks) = self.loadMasks(self.maskPath)
    #end

    def loadMasks(self, path):
        srcMasks = []
        pointMasks = []
        fileList = os.listdir(path)
        
        for i,filename in enumerate(fileList):
            splitedName = filename.split(".")
            name = splitedName[0]
            ext = splitedName[1]
            
            if "csv" == ext:
                with open(path + filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    src_pts = []
                    for i, row in enumerate(csv_reader):
                        try:
                            src_pts.append(np.array([float(row[1]), float(row[2])]))
                        except ValueError:
                            continue
                src_pts = np.array(src_pts, dtype="float32")
                pointMasks.append(src_pts)
                
                resPth = path + name + ".png"
                mask_img = cv2.imread(resPth , cv2.IMREAD_UNCHANGED)
                mask_img = mask_img.astype(np.float32) / 255.0
                srcMasks.append(mask_img)
        return srcMasks, pointMasks
    
    def mask(self, fr):
        frame = []
        face = []
        
        frame = fr.copy()
        bboxes = []
        
        faceLoc, probs, landmarks = self.detector.detect(frame, landmarks=True)
        
        for rect in faceLoc:
            (x1, y1, w, h) = rect
            (x1, y1, w, h) = int(abs(x1)), int(abs(y1)), int(abs(w)), int(abs(h))
            bbox = (x1, y1, w, h)
            bboxes.append(bbox)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            for (i, bbox) in enumerate(bboxes):
                rec = dlib.rectangle(bbox[0],bbox[1], bbox[2], bbox[3])
                
                landmark = []
                landmark = self.pose_predictor(gray, rec)
                '''
                for n in range(0,68):
                    xx = landmark.part(n).x
                    yy = landmark.part(n).y
                    result = cv2.circle(result, (xx, yy), 2, (255, 0, 255), 1)
                '''
                dst_pts = np.array( 
                    [
                        [landmark.part(1).x, landmark.part(1).y],
                        [landmark.part(2).x, landmark.part(2).y],
                        [landmark.part(3).x, landmark.part(3).y],
                        [landmark.part(4).x, landmark.part(4).y],
                        [landmark.part(5).x, landmark.part(5).y],
                        [landmark.part(6).x, landmark.part(6).y],
                        [landmark.part(7).x, landmark.part(7).y],
                        [landmark.part(8).x, landmark.part(8).y],
                        [landmark.part(9).x, landmark.part(9).y],
                        [landmark.part(10).x, landmark.part(10).y],
                        [landmark.part(11).x, landmark.part(11).y],
                        [landmark.part(12).x, landmark.part(12).y],
                        [landmark.part(13).x, landmark.part(13).y],
                        [landmark.part(14).x, landmark.part(14).y],
                        [landmark.part(15).x, landmark.part(15).y],
                        [landmark.part(29).x, landmark.part(29).y],
                    ],
                    dtype="float32",
                )
                
                if landmark != []:
                    
                    for i in range(0,len(self.pointMasks)):
                        frameC = fr.copy()
                        frameC = frameC.astype(np.float32) / 255.0
                        
                        M, _ = cv2.findHomography(self.pointMasks[i], dst_pts)
                        
                        warpedMask = cv2.warpPerspective(self.srcMasks[i], M ,(frameC.shape[1], frameC.shape[0]),None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,)
                        
                        #Maskeyi yüze yerleştirme 
                        alpha = warpedMask[:, :, 3]
                        opaklık = 1.0 - alpha #Alpha 0-1 arasında, değer
                        
                        for c in range(0, 3):
                            frameC[:, :, c] = (
                                alpha * warpedMask[:, :, c] + opaklık * frameC[:, :, c] 
                            )
                        face.append(np.array(frameC))
                        frameC = []        
        except:
            print("Hata: faceDetect")
            face = []
            face.append(np.array(frame))
            
        return face
    #end

#end::class