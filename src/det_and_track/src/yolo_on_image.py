#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

class ImageProcessor:
    def __init__(self):
        rospy.init_node('yolo_processor', anonymous=True)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.track_history = defaultdict(lambda: [])
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.image_pub = rospy.Publisher("/yolo/bounding_boxes_image", Image, queue_size=10)
    
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Printing error for line 24")
            print(e)
        
        results = self.model.track(cv_image, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot()
        print("B, TIDs:", boxes, track_ids)
        print('\n')
        print('\n')
        print('\n')
        print("Results:", results)
        print("DOne printing results")
        print('\n')
        print('\n')
        print("Printing boxes")
        print(results[0].boxes)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8"))
        except CvBridgeError as e:
            print("Printing error for line 35")
            print(e)


if __name__ == '__main__':
    ip = ImageProcessor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()