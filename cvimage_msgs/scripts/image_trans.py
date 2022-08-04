#!/usr/bin/env python 

import rospy
from std_msgs.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import pyrealsense2 as rs
from cvimage_msgs.msg import CvImage

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
            distance_mm = depth_image[point_y, point_x]

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            cv2.circle(color_image, (point_x, point_y), 8, (0, 0, 255), -1)
            cv2.putText(color_image, "{} mm".format(distance_mm), (point_x, point_y- 10), 0, 1, (0, 0, 255), 2)
            
            cv2.imshow("1", color_image)
            cv2.imshow("2", depth_colormap)

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

    point_x, point_y = 250, 100

    TransferImage()