from ultralytics import YOLO
from scipy.spatial.distance import cdist

class ObjectDetectorModel():
    def __init__(self):
        self.model_path = 'artifact/yolov8n.pt'
        self.obj_detection_model = YOLO(
            self.model_path
        )

    def detect_objects(self, image):
        results = self.obj_detection_model(image, conf=0.65, classes=[1, 2, 3, 5, 7])
        return results

    def get_class_name(self, class_id):
        return self.obj_detection_model.names[class_id]