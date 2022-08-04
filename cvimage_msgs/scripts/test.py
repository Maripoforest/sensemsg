#!/usr/bin/env python 

import rospy
from std_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pyrealsense2 as rs
from cvimage_msgs.msg import CvImage

net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                   "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
knt = 0

class TransferImage():

    # def callback_raw_image(self, data):
        
        # bridge = CvBridge()
        # cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # size = cv_image.shape[0] * cv_image.shape[1] * cv_image.shape[2]
        # data = list(cv_image.reshape(size))

        
    #     self.cvImage.data = data
    #     self.cvImage.size = list(cv_image.shape)
    #     self.cvImage.time = rospy.get_time()
    #     self.pub.publish(self.cvImage)

        # cv2.imshow("image window", cv_image)
        # cv2.waitKey(0)


    def __init__(self):

        global knt
        # point_x, point_y = 250, 100
        rospy.init_node('niryo_image_node', anonymous=True) 
        self.cvImage = CvImage()
        self.pub = rospy.Publisher('/camera_image', CvImage, queue_size=10)
        # rospy.Subscriber('/gazebo_camera/image_raw', Image, self.callback_raw_image)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            # print(depth_image)
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            blob = cv2.dnn.blobFromImage(color_image, swapRB=True)
            net.setInput(blob)
            boxes, masks = net.forward(["detection_out_final", "detection_masks"])
            detection_count = boxes.shape[2]
            # print(detection_count)

            # Boxes and masks
            # for i in range(detection_count):
            box = boxes[0, 0, 1]
            class_id = box[1]
            score = box[2]
            height, width, _ = color_image.shape

            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)

            point_x = int((x+x2)/2)
            point_y = int((y+y2)/2)

            distance_mm = float(depth_image[point_y, point_x] / 1000)
            cv2.circle(color_image, (point_x, point_y), 8, (0, 0, 255), -1)
            cv2.putText(color_image, "{} m".format(distance_mm), (point_x, point_y - 10), 0, 1, (0, 0, 255), 2)
            # roi = color_image[y: y2, x: x2]
            # roi_height, roi_width, _ = roi.shape

            # mask = masks[i, int(class_id)]
            # mask = cv2.resize(mask, (roi_width, roi_height))
            # _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

            # colors = visualize.random_colors(80)
            # contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # for cnt in contours:
                # cv2.polylines(color_image, [cnt], True, colors[i], 2)
                # img = visualize.draw_mask(color_image, [cnt], colors[i])

            cv2.rectangle(color_image, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("Img", color_image)
            # cv2.imshow("Mask", mask)
            key = cv2.waitKey(1)
            if (key == 27):
                pipeline.stop()
                break
        
            knt += 1
            knt %= 2

            if knt == 0:

                color_image = cv2.resize(color_image, (200, 150))
                cv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                size = cv_image.shape[0] * cv_image.shape[1] * cv_image.shape[2]
                data = list(cv_image.reshape(size))
                
                self.cvImage.data = data
                self.cvImage.size = list(cv_image.shape)
                self.cvImage.time = rospy.get_time()

                self.pub.publish(self.cvImage)

            # rate.sleep()

        # rospy.spin()



if __name__ == "__main__":

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # config.enable_stream(rs.stream.color, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    TransferImage()







