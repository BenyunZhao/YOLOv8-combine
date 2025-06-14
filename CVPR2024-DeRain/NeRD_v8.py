#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   NeRD_v8.py
@Time      :   2024/04/17 19:48:50
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   无
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage import img_as_ubyte
from model import MultiscaleNet as mynet

# from model_S import MultiscaleNet as myNet
import utils
from layers import *

import sys

sys.path.append("/Github/YOLOv8-Magic/ultralytics-8.1.0")
from ultralytics import YOLO


def process_image(input_image, weights_path, win_size=256, gpu_device="0"):
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
    torch.cuda.set_device(int(gpu_device))

    # 加载模型
    model_restoration = mynet()
    utils.load_checkpoint(model_restoration, weights_path)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # 处理图像
    with torch.no_grad():
        input_image = input_image.cuda()
        _, _, Hx, Wx = input_image.shape
        input_re, batch_list = window_partitionx(input_image, win_size)
        restored = model_restoration(input_re)
        restored = window_reversex(restored[0], win_size, Hx, Wx, batch_list)
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored_img = img_as_ubyte(restored[0])  # 假设直接处理的批量大小为1

    return restored_img


def main():
    # 定义图像和模型权重的路径
    input_image_path = (
        "2.png"
    )
    weights_path = "/Github/NeRD-Rain/model_large_Rain200L.pth"

    # 加载并预处理图像
    img = Image.open(input_image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)  # 添加批次维度

    # 去雨
    derained_image = process_image(input_tensor, weights_path)
    derained_pil = Image.fromarray(derained_image)

    # 保存或将处理后的图像传递给YOLO
    derained_pil.save("temp_derained_image2.jpg")

    # 加载YOLO模型
    model = YOLO("/Github/YOLOv8-Magic/ultralytics-8.1.0/yolov8n.pt")

    # 在处理后的图像上运行预测
    model.predict(
        source="temp_derained_image2.jpg",
        save=True,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        show=False,
        project="runs/predict",
        name="exp",
        save_txt=False,
        save_conf=True,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        line_width=3,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
        show_boxes=True,
    )


if __name__ == "__main__":
    main()
