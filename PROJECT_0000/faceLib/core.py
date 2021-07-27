# %%
import os
import cv2
from imutils import paths

class Core(object):

    def __init__(self, source=None):
            
            self.imagePaths =[]
            self.names = []
            self.labels = []
            self.imageNames = []
            
            self.dbPath = "./"
            self.streamSource = source
            self.vc = []
            
            if None != source:
                self.streamSource = source
                self.createVideoCapture()
            else:
                self.streamSource = 0   
    #end

    #Kamera veya Stream Oluştur
    def createVideoCapture(self, streamURL=None, w=1280, h=720, fps=60):
        if streamURL == None:
            streamURL= self.streamSource
        try:
            vc = cv2.VideoCapture(streamURL)
            vc.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            vc.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            vc.set(cv2.CAP_PROP_FPS, fps)
            vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '5'))
            self.vc = vc
            return vc
        except NameError:
            print("createVideoCaptureError"+NameError)
    #end

    #Resimler, yüz ismindeki klasörlerde tutulmalıdır. Bir klasörde birden fazla aynı kişiye ait yüz olabilir.
    
    def getFacePathNames(self, path = None):
        try:
            if path == None:
                path = self.dbPath
            #end
            
            imagePaths = list(paths.list_images(path))
            imageNames = os.listdir(path)

            names = []
            for (i, imagePath) in enumerate(imagePaths):
                names.append(imagePath.split(os.path.sep)[-2]); #Resim yollarından dosya isimleri çıkartılır.

            labels = []
            for k,name in enumerate(imageNames):
                indis = [i for i, x in enumerate(names) if x == name]
                for l in range(len(indis)):
                    labels.append(k)
            
            self.imagePaths = imagePaths
            self.names = names
            self.labels = labels
            self.imageNames = imageNames
        except:
            pass
        
        return self.imagePaths, self.names, self.labels, self.imageNames
    #end
#end::class