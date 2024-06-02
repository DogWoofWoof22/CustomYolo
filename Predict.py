from Model import Model

TRAINED_MODEL_PATH = "./runs/detect/smallModel100Epochs/weights/best.pt" #"yolov8m.pt"

TEST_PATH = "./dataset/images/test"

if __name__ == '__main__':
    model = Model(TRAINED_MODEL_PATH)
    model.predict(TEST_PATH)