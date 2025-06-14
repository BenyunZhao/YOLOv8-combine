#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File      :   sci_v8.py
@Time      :   2024/04/02 17:56:10
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   SCI + YOLOv8 联调推理脚本
'''


import sys
sys.path.append("/YOLOv8-Magic/ultralytics-8.1.0")
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
import argparse

from ultralytics import YOLO
from model import Finetunemodel

def save_image(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    im = Image.fromarray(image_numpy)
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(90, expand=True)
    im.save(path, 'png')

def main(args):
    # 加载 SCI 模型
    sci_model = Finetunemodel(args.sci_model_path)
    sci_model = sci_model.cuda()
    sci_model.eval()

    # 加载 YOLOv8 模型
    yolo_model = YOLO(args.yolo_model_path)

    # 加载输入图像
    input_image = Image.open(args.image_path).convert('RGB')
    input_tensor = torch.unsqueeze(torch.transpose(torch.tensor(np.array(input_image), dtype=torch.float32) / 255.0, 0, 2), 0).cuda()
    input_var = Variable(input_tensor, volatile=True)

    # SCI 模型的前向传播
    with torch.no_grad():
        _, enhanced_image = sci_model(input_var)

    # 保存增强后的图像
    enhanced_image_path = args.save_path
    save_image(enhanced_image, enhanced_image_path)

    # 对增强后的图像进行目标检测
    yolo_model.predict(
        source=enhanced_image_path,
        save=args.save_results,
        imgsz=args.img_size,
        conf=args.confidence,
        iou=args.iou,
        project=args.project,
        name=args.name,
        save_txt=False,
        save_conf=True,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        vid_stride=1,
        line_width=1,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
        boxes=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCI with YOLOv8 Integration by WangQvQ")
    parser.add_argument('--image_path', type=str, default='2015_06299.jpg', help='Path to the input image')
    parser.add_argument('--save_path', type=str, default='enhanced_image.png', help='Path to save the enhanced image')
    parser.add_argument('--sci_model_path', type=str, default='./weights/medium.pt', help='Path to the SCI model weights')
    parser.add_argument('--yolo_model_path', type=str, default="./yolov8n.pt", help='Path to the YOLOv8 model weights')
    parser.add_argument('--save_results', default=True, help='Whether to save the detection results')
    parser.add_argument('--img_size', type=int, default=640, help='Size of the input image')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold for object detection')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for non-maximum suppression')
    parser.add_argument('--project', type=str, default='runs/predict', help='Project name')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--line_width', default=1, help='box line width')
    args = parser.parse_args()

    main(args)
