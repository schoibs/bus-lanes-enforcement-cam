import cv2
from lane_detector import UltraFastLaneDetector
from object_detector import ObjectDetectorModel

class DashCamFootageProcessor():
    def __init__(self):
        self.lane_detector = UltraFastLaneDetector()
        self.object_detector = ObjectDetectorModel()

    def process_frame(self, image):
        # lane detection model result
        lanes_points, lanes_detected = self.lane_detector.detect_lanes(image)

        # object detection model result
        obj_detection_results = self.object_detector.detect_objects(image)
        boxes = obj_detection_results[0].boxes

        # process lanes
        first_lane = lanes_points[0]
        second_lane = lanes_points[1]

        print(f"{second_lane=}")

        most_right_coordinate = max(second_lane, key=lambda x: x[0])
        if first_lane:
            most_left_coordinate = min(first_lane, key=lambda x: x[0])
        else:
            most_left_coordinate = [most_right_coordinate[0]-200, most_right_coordinate[1]]

        if len(boxes) > 0:
            # x1, y1, x2, y2 (top-left, bottom-right corners)
            xyxy = boxes.xyxy.cpu().numpy()

            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[i]

                if (most_left_coordinate[0] <= x2 and most_right_coordinate[0] - 50 >= x1 and y1 < most_right_coordinate[1]):
                    confidence = conf[i]
                    class_id = int(cls[i])
                    class_name = self.object_detector.get_class_name(class_id)

                    # draw box
                    cv2.rectangle(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2
                    )

                    # add label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(
                        image,
                        label,
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )

        # Draw depth image
        visualization_img = self.lane_detector.draw_lanes(image, lanes_points, lanes_detected)

        return visualization_img

