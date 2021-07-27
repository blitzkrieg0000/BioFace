from abc import abstractclassmethod, ABC

class IRecognize(ABC):
    @abstractclassmethod
    def train(self):
        """ imagePaths -> tahmin, img """
        #imagePaths: Her bir klasör içerisinde yüz resmi olacak, klasör isimleri kullanıcı isimleridir.
        pass
    
   
    @abstractclassmethod
    def faceRecognize(self):
        """ boxes, img -> tahmin, img """
        
        pass