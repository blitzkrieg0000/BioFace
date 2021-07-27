from .IRecognize import IRecognize
from .addMask import AddMask
from imutils import paths
import numpy as np
import pickle
import tensorflow.keras.backend as kB
import cv2
import os
os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))


class FaceRecognizeOnnxVGG(IRecognize, AddMask):
    def __init__(self, detectFunc, thres, dataPath="./helper/trainImages"):
        super().__init__(maskPath="../helper/mask/")

        # Args
        self.detectFunc = detectFunc
        self.thres = thres
        self.cropedFaces = "yuzler/"
        (self.imagePaths, self.names, self.labels,self.imageNames) = detectFunc.getFacePathNames(dataPath)

        # Models
        self.net = cv2.dnn.readNetFromONNX('../helper/dat/vgg_model_keras_weights.onnx')
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Train
        self.encodedDB = []
        self.encodedDBLabel = []
        self.kisilerDB = []
        self.basari = 0
        try:
            print("Trained Yüklendi")
            self.data = pickle.loads(open('encodes/face_enc_VGG', "rb").read())
        except:
            print("Dosyalar bulunamadı")
            self.train(self.imagePaths, dizin="encodes/face_enc_VGG")
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
    def cropFace(self, imagePaths, dizin):
        try:
            os.mkdir(self.cropedFaces)
            print("Yüzler klasörü oluşturuldu !!")
        except:
            print("Yüzler klasörü oluşturulamadı !!")
             
        imagePaths = list(paths.list_images('../helper/trainImages'))
        for (i, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            sourceImage = cv2.imread(imagePath)

            boxes = []
            boxes, _ = self.detectFunc.faceDetect(sourceImage)
            try:
                os.mkdir(self.cropedFaces + name + "/")
            except:
                pass

            # top, right, bottom, left
            for j, (x, y, h, w) in enumerate(boxes):
                res = sourceImage[y:w, x:h]
                cv2.imwrite(self.cropedFaces + name + "/" + name +
                            str(i)+"-"+str(j) + ".jpg", res)

                images = []
                images = self.mask(sourceImage)
                for im in range(0, len(images)):
                    temp = images[im][y:w, x:h]
                    #temp = temp.astype(np.uint8)
                    temp = cv2.convertScaleAbs(temp, alpha=(255.0))
                    cv2.imwrite(self.cropedFaces + name + "/" + name + str(i)+"-" +
                                str(j) + "-" + str(im) + "_masked_" + ".jpg", temp)

    def encodeHesaplaVGG(self):
        encodedsData = []
        encodedsLabel = []
        kisiler = []
        faces = os.listdir(self.cropedFaces)
        for i, person in enumerate(faces):
            kisiler.append(person)
            goruntuler = os.listdir(self.cropedFaces + '/' + person+'/')
            for goruntu in goruntuler:

                frame = cv2.imread(self.cropedFaces + '/' + person+'/'+goruntu)

                blob = cv2.dnn.blobFromImage(cv2.resize(
                    frame, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))
                image = blob.reshape(
                    (1, blob.shape[3], blob.shape[2], blob.shape[1]))

                self.net.setInput(image)

                encode = self.net.forward()  # encoding yapılıyor 2666 feature
                encode = kB.eval(encode)

                encodedsData.append(encode)
                kisiler.append(person)
        return (encodedsData, kisiler)

    # trained data  yoksa train yapsın
    @kontrol
    def train(self, imagePaths, dizin="encodes/face_enc_VGG"):
        self.cropFace(imagePaths, self.cropedFaces)

        kisilerDB = []
        # Database encodelarını çıkart
        (encodedDB, kisilerDB) = self.encodeHesaplaVGG()

        self.encodedDB = np.array(encodedDB)
        self.kisilerDB = np.array(kisilerDB)

        self.data = {"encodings": self.encodedDB, "names": self.kisilerDB} # Kaydet
        
        f = open("encodes/face_enc_VGG", "wb")
        f.write(pickle.dumps(self.data))
        f.close()
        self.basari = 1
        
    # end

    def faceRecognize(self, boxes, image):
        nImage = []
        nImage = image.copy()
        
        def mesafeOklid(A, B):
            L = A - B
            L = np.sum(np.multiply(L, L))
            L = np.sqrt(L)
            return L

        for i, bbox in enumerate(boxes):

            try:
                sFrame = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            except:
                pass
            
            blob = cv2.dnn.blobFromImage(cv2.resize(
                sFrame, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))
            rblob = blob.reshape(
                (1, blob.shape[3], blob.shape[2], blob.shape[1]))

            self.net.setInput(rblob)
            encode = self.net.forward()  # encoding yapılıyor 2666 feature
            encode = kB.eval(encode)

            dizi = []

            for dat in self.data["encodings"]:
                dizi.append(mesafeOklid(np.array(dat), np.array(encode)))

            th = min(dizi)
    
            tahmin = ""
            if th*100 < self.thres:
                indis = np.where(dizi == th)
                tahmin =  self.data["names"][int(indis[0])]
            else:
                tahmin = "?"

            nImage = cv2.rectangle(nImage, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)
            nImage = cv2.putText(nImage, tahmin, (bbox[2] - 10, bbox[3] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (150, 200, 255), 2)
        return tahmin, nImage

    # end

    def test(self):

        while True:
            (success, imge) = self.detectFunc.vc.read()
            (bbox, image) = self.detectFunc.faceDetect(imge)
            if bbox != []:
                print(bbox)
                tahmin, frame = self.faceRecognize(bbox, image)

            cv2.imshow("Ekran", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.detectFunc.vc.release()
    # end

    def run2(self, img):
        tahmin = ["Bilinmiyor"]
        try:
            #(success, imge) = self.detectFunc.vc.read()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (bbox, frame) = self.detectFunc.faceDetect(img)
            #bbox = np.abs(bbox)
            tahmin, sonuc = self.faceRecognize(bbox, img)

        except:
            sonuc = img

        return tahmin, sonuc
    # end

# end::class