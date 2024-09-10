#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String  
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import tf
import time
import matplotlib.pyplot as plt
import io
from laser_geometry import LaserProjection
import message_filters
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN

class HumanTrajectoryTracker:
    def __init__(self):
        rospy.init_node('human_trajectory_tracker', anonymous=True)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.listener = tf.TransformListener()
        self.laser_projection = LaserProjection()

        # self.image_sub = message_filters.Subscriber("/debug_reprojection", Image)
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)

        self.scan_sub = message_filters.Subscriber("/scan_raw", LaserScan)
        self.plot_pub = rospy.Publisher("/trajectory_plot", Image, queue_size=100)

        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.scan_sub], 100, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        self.current_image = None
        self.annotated_frame = None
        self.current_points = []
        self.trajectories = {}
        self.velocities = {}


    
        self.Rcl = np.array([[0.998605, 0.0528031, -0.000539675],
                             [-0.0011676, 0.0118618, -0.999929],
                             [-0.0527929, 0.998534, 0.0119069]])
        self.tcl = np.array([-0.231367, 0.250736, 0.0676897]).reshape((3,1))
        self.K = np.array([[386.458, 0, 321.111],
                           [0, 386.458, 241.595],
                           [0, 0, 1]])

        self.plot_update_interval = 0.1
        self.last_plot_time = rospy.Time.now()
        rospy.Timer(rospy.Duration(self.plot_update_interval), self.timer_callback)

    def callback(self, image_msg, scan_msg):
        try:
            current_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            # store the ranges from scan_msg in a list
            self.range_data = scan_msg.ranges
            self.angle_min = scan_msg.angle_min
            self.angle_increment = scan_msg.angle_increment
            angles = np.arange(scan_msg.angle_min, scan_msg.angle_min + len(scan_msg.ranges) * scan_msg.angle_increment, scan_msg.angle_increment)
            x = scan_msg.ranges * np.cos(angles)
            y = scan_msg.ranges * np.sin(angles)
            points = np.vstack((x, y)).T  # Combine x, y to get points array
            # point_cloud = self.laser_projection.projectLaser(scan_msg)
            # self.process_data(current_image, point_cloud)
            self.process_data(current_image, points)

        except CvBridgeError as e:
            rospy.logerr(e)zz

    def process_data(self, image, points):
        # Filter and reproject points similar to the C++ version
        points = np.array(points)
        reprojected_points = self.reproject_points(points)
        self.current_image = image  # Update the current image
        self.current_points = reprojected_points  # Update the current points
        self.process_frame()

    def reproject_points(self, points):
        reprojected_points = []
        i = 0
        for point in points:
            point_l = np.array([point[0], point[1], 0]).reshape((3,1))
            point_c = np.dot(self.Rcl, point_l) + self.tcl
            if point_c[2] <= 0:
                continue
            if np.isfinite(point_c[0]) and np.isfinite(point_c[1]) and np.isfinite(point_c[2]):
                # print("Point_c after filtering:", point_c)
                uv = np.dot(self.K, point_c / point_c[2])
                u, v = uv[0, 0], uv[1, 0]
                x, y = point[0], point[1]
                if np.isfinite(u) and np.isfinite(v) and np.isfinite(x) and np.isfinite(y):
                    reprojected_points.append((u, v, x, y))
        return reprojected_points

    def process_frame(self):
        if self.current_image is None or self.current_points is None:
            print("No Image or Points")
            print("****\n\n\n****\n")
            return
        
        results = self.model.track(self.current_image, persist=True)

        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes.xyxy):
                cls_id = results[0].boxes.cls[i] 
                if cls_id == 0: 
                    print("Detected Human")
                    bbox = box.cpu().numpy().astype(int)
                    # extract track id only is results[0] is not None
                    if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
                        track_id = int(results[0].boxes.id[i])  # Track ID
                    else:
                        track_id = 0
                    print("Track ID:", track_id)
                    annotated_frame = results[0].plot()
                    self.annotated_frame = annotated_frame
                    points_in_bbox = self.filter_points_in_bbox(self.current_points, bbox)

                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = []
                        if points_in_bbox:
                            mean_point = np.mean([p[2:] for p in points_in_bbox], axis=0)
                            self.trajectories[track_id].append(mean_point.tolist())
                        continue

                    cluster_center = self.choose_optimal_cluster(points_in_bbox, track_id)
                    if cluster_center is not None:
                        # if track_id not in self.trajectories:
                        #     self.trajectories[track_id] = []
                        self.trajectories[track_id].append(cluster_center)
                        print(f"Track {track_id} updated with cluster center {cluster_center}")
                    else:
                        print(f"Track {track_id} not updated as no cluster center found in bbox")
                                 
            # self.publish_trajectory_plot()
                        
    def filter_points_in_bbox(self, p, bbox):
        x1, y1, x2, y2 = bbox
        return [p for p in self.current_points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]
    
    def choose_optimal_cluster(self, points, track_id):
        if len(points) < 5:
            return None
        # print("Points in Bbox:", points)
        # extract all 3rd and 4th value from all tuples in points
        points = np.array(points)
        # print("Shape of points:", points.shape)


        clustering = DBSCAN(eps=0.05, min_samples=5).fit(points[:, 2:])
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if not unique_labels:
            return None
        # Find the cluster whose mean is closest to the last point in the trajectory
        min_distance = float('inf')
        optimal_cluster_center = None
        last_point = np.array(self.trajectories[track_id][-1]) if track_id in self.trajectories and self.trajectories[track_id] else None
        for label in unique_labels:
            cluster_points = points[labels == label, 2:]
            cluster_center = np.mean(cluster_points, axis=0)
            if last_point is not None:
                distance = np.linalg.norm(cluster_center - last_point)
                if distance < min_distance:
                    min_distance = distance
                    optimal_cluster_center = cluster_center
            else:
                return cluster_center  
        return optimal_cluster_center


    
    def timer_callback(self, event):
        self.publish_trajectory_plot()
    
    def publish_trajectory_plot(self):
        current_time = rospy.Time.now()
        if (current_time - self.last_plot_time).to_sec() < self.plot_update_interval:
            return  # Not enough time has passed

        self.last_plot_time = current_time

        plt.figure()
        ax = plt.gca()
        for track_id, trajectory in self.trajectories.items():
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label=f"Track {track_id}")

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Trajectories of Detected Humans')
        plt.legend()

        ax.set_xlim([-5, 10])
        ax.set_ylim([-2, 15])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        plot_image = cv2.imdecode(plot_image, 1)
    
        if self.annotated_frame is not None:
            plot_image_resized = cv2.resize(plot_image, (self.annotated_frame.shape[1], self.annotated_frame.shape[0]))
            combined_image = np.hstack((self.annotated_frame, plot_image_resized))
            
            try:
                self.plot_pub.publish(self.bridge.cv2_to_imgmsg(combined_image, "bgr8"))
                print("Published plot")
            except CvBridgeError as e:
                rospy.logerr(e)
        else:
            try:
                self.plot_pub.publish(self.bridge.cv2_to_imgmsg(plot_image, "bgr8"))
                print("Published plot 219")
            except CvBridgeError as e:
                rospy.logerr(e)

    def calculate_velocities(self):
        for i in self.trajectories:
            self.velocities[i] = self.trajectories[i][-1] - self.trajectories[i][-2]


if __name__ == '__main__':
    try:
        tracker = HumanTrajectoryTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass