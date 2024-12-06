import cv2
import torch
import scipy.special
import numpy as np
import torchvision
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from scipy.spatial.distance import cdist

from ultrafastLaneDetector.model import parsingNet

lane_colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

tusimple_row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
			116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
			168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
			220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
			272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


class ModelType(Enum):
	TUSIMPLE = 0
	CULANE = 1

class ModelConfig():

	def __init__(self, model_type):

		if model_type == ModelType.TUSIMPLE:
			self.init_tusimple_config()
		else:
			self.init_culane_config()

	def init_tusimple_config(self):
		self.img_w = 1280
		self.img_h = 720
		self.row_anchor = tusimple_row_anchor
		self.griding_num = 100
		self.cls_num_per_lane = 56

	def init_culane_config(self):
		self.img_w = 1640
		self.img_h = 590
		self.row_anchor = culane_row_anchor
		self.griding_num = 200
		self.cls_num_per_lane = 18

class UltrafastLaneDetector():

	def __init__(self, model_path, model_type=ModelType.TUSIMPLE, use_gpu=False):

		self.use_gpu = use_gpu

		# Load model configuration based on the model type
		self.cfg = ModelConfig(model_type)

		# Initialize model
		self.lane_detection_model = self.initialize_model(model_path, self.cfg, use_gpu)

		# Initialize image transformation
		self.img_transform = self.initialize_image_transform()

		# object detection model
		self.obj_detection_model = YOLO('models/yolov8n.pt')

	@staticmethod
	def initialize_model(model_path, cfg, use_gpu):

		# Load the model architecture
		net = parsingNet(pretrained = False, backbone='18', cls_dim = (cfg.griding_num+1,cfg.cls_num_per_lane,4),
						use_aux=False) # we dont need auxiliary segmentation in testing


		# Load the weights from the downloaded model
		if use_gpu:
			if torch.backends.mps.is_built():
				net = net.to("mps")
				state_dict = torch.load(model_path, map_location='mps')['model'] # Apple GPU
			else:
				net = net.cuda()
				state_dict = torch.load(model_path, map_location='cuda')['model'] # CUDA
		else:
			state_dict = torch.load(model_path, map_location='cpu')['model'] # CPU

		compatible_state_dict = {}
		for k, v in state_dict.items():
			if 'module.' in k:
				compatible_state_dict[k[7:]] = v
			else:
				compatible_state_dict[k] = v

		# Load the weights into the model
		net.load_state_dict(compatible_state_dict, strict=False)
		net.eval()

		return net

	@staticmethod
	def initialize_image_transform():
		# Create transfom operation to resize and normalize the input images
		img_transforms = transforms.Compose([
			transforms.Resize((288, 800)),
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		])

		return img_transforms

	def detect_lanes(self, image, draw_points=True):

		obj_detection_results = self.obj_detection_model(image, conf=0.65, classes=[1, 2, 3, 5, 7])
		boxes = obj_detection_results[0].boxes

		# annotated_frame = obj_detection_results[0].plot()


		input_tensor = self.prepare_input(image)
		# input_tensor = self.prepare_input(annotated_frame)

		# Perform inference on the image
		output = self.inference(input_tensor)

		# Process output data
		self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)
		first_lane = self.lanes_points[0]
		second_lane = self.lanes_points[1]

		most_right_coordinate = max(second_lane, key=lambda x: x[0])
		print(f"hello {most_right_coordinate=}")
		if first_lane:
			most_left_coordinate = min(first_lane, key=lambda x: x[0])
		else:
			most_left_coordinate = [most_right_coordinate[0]-200, most_right_coordinate[1]]
			# first_lane = [[x - 200, y - 200] for x, y in second_lane]
			# most_left_coordinate = min(first_lane, key=lambda x: x[0])

		print(f"{most_right_coordinate=}") # [590, 409]
		# print(f"{self.lanes_detected=}")

		if len(boxes) > 0:
			xyxy = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format (top-left, bottom-right corners)
			# xywh = boxes.xywh.cpu().numpy()  # x, y, width, height format (center x, center y)
			# xyxyn = boxes.xyxyn.cpu().numpy()  # normalized xyxy format (values between 0-1)
			# xywhn = boxes.xywhn.cpu().numpy()

			# print(f"{xyxy=}")
			# print(f"{xywh=}")
			# print(f"{xyxyn=}")
			# print(f"{xywhn=}")
			# print(f"{xyxy=}")

			conf = boxes.conf.cpu().numpy()
			cls = boxes.cls.cpu().numpy()

			for i in range(len(boxes)):
				x1, y1, x2, y2 = xyxy[i]

				# if (x2 < most_right_coordinate[0]) and (y1 < most_right_coordinate[1]):
				if (most_left_coordinate[0] <= x2 and most_right_coordinate[0] - 50 >= x1 and y1 < most_right_coordinate[1]):
				# if any(min <= x2 and max >= x1 for (max, min), (_, _) in zip(second_lane, first_lane)) and y1 < most_right_coordinate[1]:

					confidence = conf[i]
					class_id = int(cls[i])
					class_name = self.obj_detection_model.names[class_id]

					# Draw box
					cv2.rectangle(image,
								(int(x1), int(y1)),
								(int(x2), int(y2)),
								(0, 255, 0), 2)

					# Add label
					label = f"{class_name}: {confidence:.2f}"
					cv2.putText(image, label,
							(int(x1), int(y1)-10),
							cv2.FONT_HERSHEY_SIMPLEX,
							0.5, (0, 255, 0), 2)

		# Draw depth image
		visualization_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, self.cfg, draw_points)
		# visualization_img = self.draw_lanes(annotated_frame, self.lanes_points, self.lanes_detected, self.cfg, draw_points)

		return visualization_img

	def prepare_input(self, img):
		# Transform the image for inference
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_pil = Image.fromarray(img)
		input_img = self.img_transform(img_pil)
		input_tensor = input_img[None, ...]

		if self.use_gpu:
			if not torch.backends.mps.is_built():
				input_tensor = input_tensor.cuda()

		return input_tensor

	def inference(self, input_tensor):
		with torch.no_grad():
			output = self.lane_detection_model(input_tensor)

		return output

	@staticmethod
	def process_output(output, cfg):
		# Parse the output of the model
		processed_output = output[0].data.cpu().numpy()
		processed_output = processed_output[:, ::-1, :]
		prob = scipy.special.softmax(processed_output[:-1, :, :], axis=0)
		idx = np.arange(cfg.griding_num) + 1
		idx = idx.reshape(-1, 1, 1)
		loc = np.sum(prob * idx, axis=0)
		processed_output = np.argmax(processed_output, axis=0)
		loc[processed_output == cfg.griding_num] = 0
		processed_output = loc


		col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
		col_sample_w = col_sample[1] - col_sample[0]

		lanes_points = []
		lanes_detected = []

		max_lanes = processed_output.shape[1]
		for lane_num in range(max_lanes):
			lane_points = []
			# Check if there are any points detected in the lane
			if np.sum(processed_output[:, lane_num] != 0) > 2:

				lanes_detected.append(True)

				# Process each of the points for each lane
				for point_num in range(processed_output.shape[0]):
					if processed_output[point_num, lane_num] > 0:
						lane_point = [int(processed_output[point_num, lane_num] * col_sample_w * cfg.img_w / 800) - 1, int(cfg.img_h * (cfg.row_anchor[cfg.cls_num_per_lane-1-point_num]/288)) - 1 ]
						lane_points.append(lane_point)
			else:
				lanes_detected.append(False)

			lanes_points.append(lane_points)
		return np.array(lanes_points, dtype="object"), np.array(lanes_detected)

	@staticmethod
	def draw_lanes(input_img, lanes_points, lanes_detected, cfg, draw_points=True):
		# Write the detected line points in the image
		visualization_img = cv2.resize(input_img, (cfg.img_w, cfg.img_h), interpolation = cv2.INTER_AREA)

		# Draw a mask for the current lane
		if(lanes_detected[1] and lanes_detected[2]):
			lane_segment_img = visualization_img.copy()

			# cv2.fillPoly(lane_segment_img, pts = [np.vstack((lanes_points[1],np.flipud(lanes_points[2])))], color =(255,191,0))
			visualization_img = cv2.addWeighted(visualization_img, 0.7, lane_segment_img, 0.3, 0)

		if(draw_points):
			for lane_num,lane_points in enumerate(lanes_points):
				for lane_point in lane_points:
					cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)

		return visualization_img










