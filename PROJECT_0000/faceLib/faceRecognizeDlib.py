from .IRecognize import IRecognize
from .addMask import AddMask
import numpy as np
import pickle
import dlib
import cv2
import os
import time
from tqdm import tqdm

os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))


class FaceRecognizeDlib(IRecognize, AddMask):

    def __init__(self, detectFunc, thres, dataPath="../helper/trainImages"):
        super().__init__(maskPath="../helper/mask/")

        # Args
        self.detectFunc = detectFunc
        self.thres = thres
        (self.imagePaths, self.names, self.labels, self.imageNames) = detectFunc.getFacePathNames(dataPath)
        self.knownNames = []
        
        # Models
        self.pose_predictor = dlib.shape_predictor("../helper/dat/shape_predictor_68_face_landmarks.dat")
        self.face_encoder = dlib.face_recognition_model_v1('../helper/dat/dlib_face_recognition_resnet_model_v1.dat')
        self.basari = 0
        # Train
        try:
            self.data = pickle.loads(open('encodes/face_enc_dlib', "rb").read())
            print("Eğitilmiş Veri Yüklendi! \n")
        except:
            print("Eğitilmiş Veri Bulunamadı! \n")
            self.train(self.imagePaths, dizin="encodes/face_enc_dlib")
    # end

    def kontrol(func):
        def inner(self, imagePaths, dizin):
            if not os.path.exists(dizin):
                return func(self, imagePaths, dizin)
            else:
                print("< " + str(dizin) +  " > dizini var olduğundan çalıştırılmadı!")
                return
        return inner
    # end
    
    @kontrol
    def train(self, imagePaths, dizin):
        print("\n\n\n\n\n")
        knownEncodings = []
        knownNames = []
        
        xa = range(len(imagePaths))
        ya = range(200)
        total = min(len(xa), len(ya))
        
        for (k, imagePath) in tqdm(enumerate(imagePaths), total=total, desc ="Eğitiliyor...", colour="#eb6734"):
            imge = cv2.imread(imagePath)
                
            box = []
            (box, _) = self.detectFunc.faceDetect(imge)

            if [] != box:
                rect = box[0]
                images = []
                images = self.mask(imge)
                images.append(imge)      #Maskeli ve Orijinal Resimler
                
                for img in images:
                    img = cv2.convertScaleAbs(img, alpha=(255.0))
                    landmark = self.pose_predictor(cv2.cvtColor( img, cv2.COLOR_BGR2GRAY), dlib.rectangle(rect[0], rect[1], rect[2], rect[3]))
                    rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = self.face_encoder.compute_face_descriptor(rgbImg, landmark, num_jitters=10)
                        
                    knownNames.append(self.names[k])
                    knownEncodings.append(encodings)
                # end
                    
            else:
                print("\n Başarısız: " + imagePath + "\n")
                #knownNames.append(self.names[k])
                #knownEncodings.append()
                    
        
        self.knownNames = knownNames
        self.data = {"encodings": knownEncodings, "names": knownNames} # Kaydet
        
        f = open("encodes/face_enc_dlib", "wb")
        f.write(pickle.dumps(self.data))
        f.close()
        self.basari = 1
    # end
    
    def faceRecognize(self, boxes, img):
        imgC = []
        imgC = img.copy()

       # İki nokta arasındaki mesafe // A(x1, x2, ..., xn) <---d---> B(y1, y2, ..., yn)
        def mesafeOklid(A, B): 
            L = A - B
            L = np.sum(np.multiply(L, L))
            L = np.sqrt(L)
            return L

        for bbox in boxes:
            
            gray = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)
            rec = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])

            landmark = self.pose_predictor(gray, rec)
            rgbImg = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)
            encoding = self.face_encoder.compute_face_descriptor(rgbImg, landmark, num_jitters=10)
            dizi = []
            for dat in  self.data['encodings']:
                dizi.append(mesafeOklid(np.array(dat), np.array(encoding)))
            #dizi = np.linalg.norm(data["encodings"] - np.array(encodings), axis=1)
            th = min(dizi)
            
            tahmin = ""
            if th*100 < self.thres:
                indis = np.where(dizi == th)
                tahmin =  self.data["names"][int(indis[0])]
            else:
                tahmin = "?"

            imgC = cv2.putText(imgC, tahmin, (bbox[2] - 10, bbox[3] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 200, 255), 2)
        return tahmin, imgC
    # end

    def test(self):

        while True:
            (success, frame) = self.detectFunc.vc.read()
            tahmin = ["Bilinmeyen"]
            sonuc = []
            
            try:
                (bbox, frame) = self.detectFunc.faceDetect(frame)
                tahmin, frame = self.faceRecognize(bbox, frame)
            except:
                pass

            cv2.imshow("Ekran", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.detectFunc.vc.release()
    # end

    def run2(self, img):
        tahmin = ["Bilinmeyen"]
        sonuc = []
        try:
            #(success, imge) = self.detectFunc.vc.read()
            (bbox, frame) = self.detectFunc.faceDetect(img)
            tahmin, sonuc = self.faceRecognize(bbox, frame)
        except:
            sonuc = img
        return tahmin, sonuc
    # end

# end::class
