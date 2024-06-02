from ultralytics import YOLO

class Model:
    def __init__(self,YoloVersion) -> None:
        self.model = YOLO(YoloVersion)
        
    def train(self,data,epochs):
        self.model.train(data=data, epochs=epochs, imgsz=640)
        
    def predict(self,imgPath):
        self.model.predict(imgPath, save=True, imgsz=640)