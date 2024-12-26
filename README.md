# Bus Lanes Enforcement Camera
This project aims to develop a prototype AI camera system for automated bus lane enforcement. The goal is to protect dedicated bus lanes from unauthorized vehicle use.

We propose leveraging the widespread adoption of car dashboard cameras (dashcams) as enforcement tools. The idea is to have many of these individual dashcam cameras to act as enforcement cameras, creating a distributed network of monitoring devices. More eyes on the road means greater enforcement of traffic rules. Hence, this prototype uses sample videos that simulate typical dashcam footage.

# Prerequisite

### Libraries
```shell
pip3 -m venv venv
source venv/bin/activate
pip install -r requirements
```

### Pretrained model
1. Create a folder `artifact`
2. Download the pretrained Res-18 model on Tusimple dataset from the [original UFLD repository](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and save it into the `artifact` folder.

### Output
1. Create a folder `output`

# Running the Project

 ```shell
 python main.py
 ```

![screenshot_1](media/screenshot_1.png)
![screenshot_2](media/screenshot_2.png)
![screenshot_3](media/screenshot_3.png)

# Reference

- the model used for lane detections: [original UFLD repository](https://github.com/cfzd/Ultra-Fast-Lane-Detection)
- this project's code draws extensively from this repository [ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/tree/main)
- the sample video is taken from this [youtube vid](https://www.youtube.com/watch?v=qCW_hJTGTLQ&t=2s)
