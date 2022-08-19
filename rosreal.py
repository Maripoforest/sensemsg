#!/usr/bin/env python
import rospy
import os, sys
import cv2
import pyrealsense2 as rs
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math
from cvimage_msgs.msg import CvImage
from cvimage_msgs.srv import *

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

pub_color = rospy.Publisher('/camera_image_color', CvImage, queue_size=10)
pub_depth = rospy.Publisher('/camera_image_depth', CvImage, queue_size=10)
cvImage_color = CvImage()
cvImage_depth = CvImage()
P = [0.0, 0.0, 0.0]


def send_position(p, n):
        # rospy.wait_for_service('/pick_litter')
        try:
            send = rospy.ServiceProxy('pick_litter', Litter)
            response = send(p[0],p[2],p[1],n)
            return response.rst
        except Exception as e:
            print(e)

@torch.no_grad()
def run():
    global cvImage_color, cvImage_depth, P
    knt = 0
    weights='best.pt'  # model.pt path(s)
    imgsz=640  # inference size (pixels)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=10  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    stride = 32
    device_num='cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    update=False  # update all models
    name='exp'  # save results to project/name

    # Initialize
    set_logging()
    device = select_device(device_num)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location = device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location = device)['model']).to(device).eval()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    while(True):
    
        t0 = time.time()

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # check for common shapes
        s = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in img], 0)  # shapes
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=augment,
                     visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, img0)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    xc = int((xyxy[0] + xyxy[2])/2)
                    yc = int((xyxy[1] + xyxy[3])/2)
                    distance_mm = depth_image[yc, xc]
                    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
                    p = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [xc, yc], distance_mm)
                    rst = 
                    if p[0] != 0.0:
                        rst = send_position(p, names[c])
                    for i in range(3):
                        f = int(p[i])
                        cv2.putText(img0, "{} mm".format(f), (xc, yc - 10 - i*15), 0, 0.6, (0, 0, 255), 2)
                    cv2.circle(img0, (xc, yc), 8, (0, 0, 255), -1)
                    # cv2.putText(img0, "{} mm".format(p), (xc, yc - 10), 0, 1, (0, 0, 255), 1)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                   

        cv2.imshow("IMAGE", img0)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        # cv2.imshow("DEPTH", depth_colormap)
        knt += 1
        knt %= 2
        if knt == 0:
            depth_colormap = cv2.resize(depth_colormap, (200,150))
            size = depth_colormap.shape[0] * depth_colormap.shape[1] * depth_colormap.shape[2]
            data = list(depth_colormap.reshape(size))
            cvImage_depth.time = rospy.get_time()
            cvImage_depth.data = data
            cvImage_depth.size = list(depth_colormap.shape)
            pub_depth.publish(cvImage_depth)
            # img0 = cv2.resize(img0, (200,150))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

if __name__ == '__main__':
    rospy.init_node("realsense")
    run()
