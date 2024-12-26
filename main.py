import cv2
from model import DashCamFootageProcessor

# lane detection model
processor = DashCamFootageProcessor()

# open sample video
cap = cv2.VideoCapture("input/sample_input.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# save to output folder
out = cv2.VideoWriter("output/sample_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

cv2.namedWindow("Dashcam Camera", cv2.WINDOW_NORMAL)

while cap.isOpened():
	try:
		ret, frame = cap.read()
	except:
		continue

	if ret:
		output_img = processor.process_frame(frame)

		cv2.imshow("Dashcam Camera", output_img)
		out.write(output_img)
	else:
		break

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()