# pose_webcam.py
# Modified from the code provided here: https://github.com/Microsoft/human-pose-estimation.pytorch/issues/26#issuecomment-447404791
import argparse
import os
import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.inference import get_final_preds, get_max_preds
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger
from utils.transforms import *
import cv2
import dataset
import models
import numpy as np
from PIL import Image
from mss import mss

def main():
    ###mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
    ###sct = mss()
    # Specify the configuration file and model file for the neural network.
    update_config("experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml")
    config.TEST.MODEL_FILE = "models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar"
    ###cap = cv2.VideoCapture(-1)
    cap = cv2.VideoCapture('vid6.mp4')  # Load the video


    # Configure some settings for CUDA.
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )

    # Load the model for the neural network.
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Switch to evaluate mode (not training mode)
    model.eval()
    with torch.no_grad():
        while True:
            ###img = sct.grab(mon)
            ###cap = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            ret, data_numpy = cap.read()  # Read an image from the frame.
            # data_numpy.namedWindom("frame" , 0)
            # cv2.resizeWindow("frame", 800,600);

            # data_numpy = cv2.imread(frame, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            ###model_input = cv2.resize(data_numpy, (192, 256))
            model_input = cv2.resize(data_numpy, (192, 256), fx=0, fy=0, interpolation = cv2.INTER_NEAREST)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
            model_input = transform(model_input).unsqueeze(0)

            # Compute output heatmap
            output = model(model_input)
            # Compute coordinates of the joints.
            output_matrix = output.clone().cpu().numpy()
            preds, maxvals = get_max_preds(output_matrix)
            # Display the points on the screen (rescaled to the original image dimensions).
            image = data_numpy.copy()
            y_scale, x_scale = image.shape[0]/output_matrix.shape[2], image.shape[1]/output_matrix.shape[3]

            keyp = []

            right_shoulder = preds[0][6]
            x,y = int(right_shoulder[0] * x_scale), int(right_shoulder[1] * y_scale)
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            keyp.append(right_shoulder)

            left_shoulder = preds[0][5]
            x,y = int(left_shoulder[0] * x_scale), int(left_shoulder[1] * y_scale)
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            keyp.append(left_shoulder)

            left_hip = preds[0][11]
            x,y = int(left_hip[0] * x_scale), int(left_hip[1] * y_scale)
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            keyp.append(left_hip)

            right_hip = preds[0][12]
            x,y = int(right_hip[0] * x_scale), int(right_hip[1] * y_scale)
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)
            keyp.append(right_hip)

            center = sum(keyp)

            x, y = int(center[0]/4 * x_scale), int (center[1]/4 * y_scale)
            cv2.circle(image, (x, y), 5, (0, 255, 0), 5)

            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            cv2.imshow('Visualisation', image)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
