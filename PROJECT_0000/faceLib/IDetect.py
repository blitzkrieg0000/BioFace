from abc import abstractclassmethod, ABC

class IDetect(ABC):
    @abstractclassmethod
    def faceDetect(self):
        """image -> bbox,face """
        pass