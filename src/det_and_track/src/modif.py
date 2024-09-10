#!/usr/bin/env python

import rospy
from std_msgs.msg import String  
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import tf
import time
import matplotlib.pyplot as plt
import io

class HumanTrajectoryTracker:
    def __init__(self):
        rospy.init_node('human_trajectory_tracker', anonymous=True)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.listener = tf.TransformListener()

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.points_sub = rospy.Subscriber("/Reprojection_pts", String, self.points_callback)  
        self.plot_pub = rospy.Publisher("/trajectory_plot", Image, queue_size=10)

        self.current_image = None
        self.current_points = []
        self.trajectories = {}
        self.last_points_timestamp = rospy.Time(0)
        self.last_processed_frame_time = rospy.Time(0) 

        self.plot_update_interval = 1  # Interval in seconds
        self.last_plot_time = rospy.Time(0)
        rospy.Timer(rospy.Duration(self.plot_update_interval), self.timer_callback)


    def image_callback(self, data):
        current_frame_time = data.header.stamp
        if self.last_points_timestamp > self.last_processed_frame_time:
            try:
                self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.process_frame()
                self.last_processed_frame_time = current_frame_time
            except CvBridgeError as e:
                rospy.logerr(e)

    def points_callback(self, msg):
        timestamp = rospy.Time.now()
        try:
            u, v, x, y = map(float, msg.data.split(','))
            self.current_points.append((u, v, x, y))
            self.last_points_timestamp = timestamp
        except ValueError as e:
            rospy.logerr(f"Error parsing point data: {e}")

    def process_frame(self):
        if self.current_image is None or self.current_points is None:
            return
        
        results = self.model.track(self.current_image, persist=True)

        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes.xyxy):
                cls_id = results[0].boxes.cls[i] 
                if cls_id == 0: 
                    print("Detected Human")
                    bbox = box.cpu().numpy().astype(int)
                    # print("Bbox:", bbox)
                    track_id = int(results[0].boxes.id[i])  # Track ID
                    print("Track ID:", track_id)
                    # print("Lidar Points:", self.current_points)
                    points_in_bbox = self.filter_points_in_bbox(self.current_points, bbox)
                    if points_in_bbox:
                        # print("Some Points detected in Bbox:", points_in_bbox)
                        # mean_point = np.mean([p[1:3] for p in points_in_bbox], axis=0)
                        mean_point = np.mean([p[2:] for p in points_in_bbox], axis=0)
                        # print("Mean Point:", mean_point)
                        if track_id in self.trajectories:
                            self.trajectories[track_id].append(mean_point.tolist())
                        else:
                            self.trajectories[track_id] = [mean_point.tolist()]
                    else:
                        print("No Points detected in Bbox")
                            
            # print("Trajectory:", self.trajectories)
            # time.sleep(1)        
            # self.publish_trajectory_plot()
                        
    def filter_points_in_bbox(self, p, bbox):
        x1, y1, x2, y2 = bbox
        return [p for p in self.current_points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
    
    def timer_callback(self, event):
        self.publish_trajectory_plot()
    
    def publish_trajectory_plot(self):
        current_time = rospy.Time.now()
        if (current_time - self.last_plot_time).to_sec() < self.plot_update_interval:
            return  # Not enough time has passed

        self.last_plot_time = current_time

        plt.figure()
        for track_id, trajectory in self.trajectories.items():
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label=f"Track {track_id}")

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Trajectories of Detected Humans')
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        plot_image = cv2.imdecode(plot_image, 1)
    
        if self.current_image is not None:
            plot_image_resized = cv2.resize(plot_image, (self.current_image.shape[1], self.current_image.shape[0]))
            combined_image = np.hstack((self.current_image, plot_image_resized))
            
            try:
                self.plot_pub.publish(self.bridge.cv2_to_imgmsg(combined_image, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(e)
        else:
            try:
                self.plot_pub.publish(self.bridge.cv2_to_imgmsg(plot_image, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(e)

        # try:
        #     self.plot_pub.publish(self.bridge.cv2_to_imgmsg(plot_image, "bgr8"))
        # except CvBridgeError as e:
        #     rospy.logerr(e)

if __name__ == '__main__':
    try:
        tracker = HumanTrajectoryTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass