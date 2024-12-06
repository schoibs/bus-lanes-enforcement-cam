import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# lane detection model
lane_model = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False
lane_detector = UltrafastLaneDetector(lane_model, model_type, use_gpu)

# open sample video
cap = cv2.VideoCapture("videos/video_3.mp4")


cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:
		# Detect the lanes
		output_img = lane_detector.detect_lanes(frame)

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()