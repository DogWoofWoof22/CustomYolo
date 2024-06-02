from Model import Model

DATA = "datasetDescriptor.yaml"
EPOCHS = 100


MODEL_VER = "yolov8s.pt" #"yolov8m.pt"


if __name__ == '__main__':
    model = Model("yolov8s.pt")
    model.train(DATA,EPOCHS)