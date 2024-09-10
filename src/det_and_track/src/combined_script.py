#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, LaserScan, PointCloud2
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
from nav_msgs.msg import Odometry
from math import atan2, degrees, radians, sqrt
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import open3d as o3d
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import cv2
import json

class HumanTrajectoryTracker:
    def __init__(self):
        rospy.init_node('human_trajectory_tracker', anonymous=True)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.listener = tf.TransformListener()
        self.laser_projection = LaserProjection()

        # Subscribers
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        self.scan_sub = message_filters.Subscriber("/scan_raw", LaserScan)

        # self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        # self.depth_sub = message_filters.TimeSynchronizer([self.depth_sub], queue_size=1)
        # self.depth_sub.registerCallback(self.depth_callback)

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        
        # Publishers
        self.plot_pub = rospy.Publisher("/trajectory_plot", Image, queue_size=100)
        # self.marker_pub = rospy.Publisher('/trajectory_points', Marker, queue_size=10)
        self.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
        
        self.trajectories_pub = rospy.Publisher('/trajectories', String, queue_size=10)
        self.velocities_pub = rospy.Publisher('/velocities', String, queue_size=10)

        
        # Initialize lists
        self.x_coords = []
        self.y_coords = []
        self.headings = []

        self.markers = {}
        self.arrow_markers = {}
        
        # Initialize the initial heading
        self.initial_heading = None
        self.current_heading = None
        
        # Synchronizer
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.scan_sub], 100, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        # Data holders
        self.current_image = None
        self.annotated_frame = None
        self.current_points = []
        self.trajectories = {}
        self.velocities = {}
        self.map_frame_trajectories = {}
        self.depth_image = None

        self.Rcl = np.array([[0.998605, 0.0528031, -0.000539675],
                             [-0.0011676, 0.0118618, -0.999929],
                             [-0.0527929, 0.998534, 0.0119069]])
        self.tcl = np.array([-0.231367, 0.250736, 0.0676897]).reshape((3,1))

        self.fx = 386.458
        self.fy = 386.458
        self.cx = 321.111
        self.cy = 241.595
        self.K = np.array([[386.458, 0, 321.111],
                           [0, 386.458, 241.595],
                           [0, 0, 1]])

        self.plot_update_interval = 0.1
        self.last_plot_time = rospy.Time.now()
        rospy.Timer(rospy.Duration(self.plot_update_interval), self.timer_callback)

    def depth_callback(self, data):
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        points = self.convert_depth_image_to_point_cloud(depth_image, factor=4)
        point_cloud_msg = self.create_point_cloud(points, data.header)
        self.point_cloud_pub.publish(point_cloud_msg)

    def convert_depth_image_to_point_cloud(self, depth_image, factor):
        height, width = depth_image.shape

        # depth_image = cv2.resize(depth_image, (width // 4, height // 4))
        depth_image = cv2.resize(depth_image, (width // factor, height // factor))

        height, width = height // factor, width // factor
        points = []

        for v in range(height):
            for u in range(width):
                z = depth_image[v, u] * 0.001  # Convert mm to meters
                if z == 0:
                    continue
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                points.append([x, y, z])

        # points = [[0, 0, 1], [1, 1, 1], [2, 2, 2]]

        # print("###########################:", points)

        return np.array(points, dtype=np.float32)

    def create_point_cloud(self, points, header):
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        pitch_angle = -30
        yaw_angle=45
        # roll_angle = -45
        roll_angle = -50

        pitch_angle = np.deg2rad(pitch_angle)
        yaw_angle = np.deg2rad(yaw_angle)
        roll_angle = np.deg2rad(roll_angle)

        R_roll = np.array([
        [np.cos(roll_angle), -np.sin(roll_angle), 0],
        [np.sin(roll_angle), np.cos(roll_angle), 0],
        [0, 0, 1]
        ])
        
        # Define rotation matrices
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
            [0, np.sin(pitch_angle), np.cos(pitch_angle)]
        ])
        
        R_yaw = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])
        
        # Combine the rotation matrices
        R = R_pitch @ R_yaw @ R_roll
        
        # Rotate the points
        rotated_points = np.dot(points, R.T)
        
        point_cloud_msg = pc2.create_cloud(header, fields, rotated_points)
        return point_cloud_msg



    def odom_callback(self, data):
        # Extract position from Odometry message
        position = data.pose.pose.position
        self.x_coords.append(position.x)
        self.y_coords.append(position.y)
        
        # Compute the heading
        if len(self.x_coords) > 1:
            dx = self.x_coords[-1] - self.x_coords[-2]
            dy = self.y_coords[-1] - self.y_coords[-2]
            heading = atan2(dy, dx)
            heading_deg = degrees(heading)
        else:
            heading_deg = 0  # No movement, no heading
        
        self.headings.append(heading_deg)
        
        # Set initial heading if not set
        if self.initial_heading is None:
            self.initial_heading = heading_deg
        
        # Compute relative heading
        relative_headings = [heading - self.initial_heading for heading in self.headings]

        self.current_heading = relative_headings[-1]
        
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
            self.process_data(current_image, points)
        except CvBridgeError as e:
            rospy.logerr(e)

    def transform_points(self, translation, angle):
        angle_rad = - radians(angle)
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        for key in self.trajectories:
            transformed_points = []
            points = self.trajectories[key]
            for point in points:
                x, y = point[0], point[1]
                x_translated, y_translated = x + translation[0], y + translation[1]
                transformed_points.append([x_translated, y_translated])

            self.map_frame_trajectories[key] = transformed_points

    def process_data(self, image, points):
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
                    if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
                        track_id = int(results[0].boxes.id[i])
                    else:
                        track_id = 0
                    print("Track ID:", track_id)
                    annotated_frame = results[0].plot()
                    self.annotated_frame = annotated_frame
                    points_in_bbox = self.filter_points_in_bbox(self.current_points, bbox)
                    if points_in_bbox:
                        mean_point = np.mean([p[2:] for p in points_in_bbox], axis=0)
                        if track_id in self.trajectories:
                            self.trajectories[track_id].append(mean_point.tolist())
                        else:
                          self.trajectories[track_id] = [mean_point.tolist()]
                    else:
                        print("No Points detected in Bbox")
                                 
        for key in list(self.trajectories):
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                if results[0].boxes.id is not None and key not in results[0].boxes.id:
                    del self.trajectories[key]

        self.calculate_velocities()
        
        self.publish_trajectories_and_velocities()

                        
    def filter_points_in_bbox(self, p, bbox):
        x1, y1, x2, y2 = bbox
        return [p for p in self.current_points if x1 <= p[0] <= x2 and y1 <= p[1] <= y2]

    def choose_optimal_cluster(self, points, track_id):
        if len(points) < 5:
            return None
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(points[:, 2:])
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if not unique_labels:
            return None
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

        valid_track_ids = set(self.trajectories.keys())


        '''
        for track_id, trajectory in self.trajectories.items():
            # Plot the last point of the trajectory
            trajectory = np.array([trajectory[-1]])
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label=f"Track {track_id}")

            # ROS Marker for the trajectory point
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "trajectory"
            marker.id = track_id
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            for point in trajectory:
                p = Point()
                p.x, p.y = point
                p.z = 0
                marker.points.append(p)

            # Create an arrow to represent velocity in ROS markers
            if track_id in self.velocities:
                velocity = self.velocities[track_id] # sqrt(self.velocities[track_id][0] ** 2 + self.velocities[track_id][0] ** 2)
                arrow_marker = Marker()
                arrow_marker.header.frame_id = "base_link"
                arrow_marker.header.stamp = rospy.Time.now()
                arrow_marker.ns = "trajectory_velocity"
                arrow_marker.id = track_id + 1000  # Unique ID for the arrow
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                arrow_marker.pose.orientation.w = atan2(self.velocities[track_id][1], self.velocities[track_id][0])
                arrow_marker.scale.x = 0.1  # Width of the arrow shaft
                arrow_marker.scale.y = 0.2  # Width of the arrow head
                arrow_marker.color.r = 0.0
                arrow_marker.color.g = 0.0
                arrow_marker.color.b = 1.0
                arrow_marker.color.a = 1.0

                # Define the start (current position) and end (based on velocity) of the arrow
                start_point = Point()
                start_point.x, start_point.y = trajectory[0]
                start_point.z = 0

                end_point = Point()
                end_point.x = start_point.x + velocity[0]
                end_point.y = start_point.y + velocity[1]
                end_point.z = 0

                arrow_marker.points.append(start_point)
                arrow_marker.points.append(end_point)

                self.arrow_markers[track_id] = arrow_marker

                self.marker_pub.publish(arrow_marker)

            # Update or publish the trajectory marker
            self.markers[track_id] = marker
            # self.marker_pub.publish(marker)

        # Clean up old markers
        for track_id in list(self.markers.keys()):
            if track_id not in valid_track_ids:
                delete_marker = Marker()
                delete_marker.header.frame_id = "base_link"
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "trajectory"
                delete_marker.id = track_id
                delete_marker.type = Marker.POINTS
                delete_marker.action = Marker.DELETE

                # self.marker_pub.publish(delete_marker)
                del self.markers[track_id]
                
        for track_id in list(self.arrow_markers.keys()):
            if track_id not in valid_track_ids:
                delete_marker = Marker()
                delete_marker.header.frame_id = "base_link"
                delete_marker.header.stamp = rospy.Time.now()
                delete_marker.ns = "trajectory"
                delete_marker.id = track_id
                delete_marker.type = Marker.ARROW
                delete_marker.action = Marker.DELETE

                self.marker_pub.publish(delete_marker)
                del self.arrow_markers[track_id]
        '''

        # Matplotlib part: Plot velocity arrows
        for track_id, trajectory in self.trajectories.items():
            if track_id in self.velocities:
                velocity = self.velocities[track_id]
                trajectory = np.array([trajectory[-1]])  # Get the latest point
                
                # Plot the marker at the trajectory point
                plt.plot(trajectory[0, 0], trajectory[0, 1], 'ro')
                
                # Plot the velocity arrow using quiver
                ax.quiver(
                    trajectory[0, 0], trajectory[0, 1],  # Starting point (x, y)
                    velocity[0], velocity[1],             # Velocity components (dx, dy)
                    angles='xy', scale_units='xy', scale=1, color='blue', label=f"Velocity {track_id}"
                )

                # Add track_id as text next to the marker
                ax.text(trajectory[0, 0] + 0.1, trajectory[0, 1] + 0.1,  # Adjust the offset for better visibility
                        f"{track_id}", fontsize=9, color='black')

        # Finalize the plot
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Trajectories of Detected Humans with Velocities')
        plt.legend()

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        # Convert the plot to an image and publish
        plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        plot_image = cv2.imdecode(plot_image, 1)

        # if self.annotated_frame is not None:
        #     plot_image_resized = cv2.resize(plot_image, (self.annotated_frame.shape[1], self.annotated_frame.shape[0]))
        #     combined_image = np.hstack((self.annotated_frame, plot_image_resized))
            
        #     try:
        #         self.plot_pub.publish(self.bridge.cv2_to_imgmsg(combined_image, "bgr8"))
        #     except CvBridgeError as e:
        #         rospy.logerr(e)
        # else:
        #     try:
        #         self.plot_pub.publish(self.bridge.cv2_to_imgmsg(plot_image, "bgr8"))
        #     except CvBridgeError as e:
        #         rospy.logerr(e)

                
        #     # for track_id in list(self.markers.keys()):
        #     #     if track_id not in valid_track_ids:
        #     #         delete_marker = Marker()
        #     #         delete_marker.header.frame_id = "base_link"
        #     #         delete_marker.header.stamp = rospy.Time.now()
        #     #         delete_marker.ns = "trajectory"
        #     #         delete_marker.id = track_id
        #     #         delete_marker.type = Marker.POINTS
        #     #         delete_marker.action = Marker.DELETE

        #     #         self.marker_pub.publish(delete_marker)
        #     #         del self.markers[track_id]

        #     plt.xlabel('X Coordinate')
        #     plt.ylabel('Y Coordinate')
        #     plt.title('Trajectories of Detected Humans')
        #     plt.legend()

        #     ax.set_xlim([-10, 10])
        #     ax.set_ylim([-10, 10])

        #     buf = io.BytesIO()
        #     plt.savefig(buf, format='png')
        #     plt.close()
        #     buf.seek(0)

        #     plot_image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        #     plot_image = cv2.imdecode(plot_image, 1)
        
        if self.annotated_frame is not None:
            plot_image_resized = cv2.resize(plot_image, (self.annotated_frame.shape[1], self.annotated_frame.shape[0]))
            combined_image = np.hstack((self.annotated_frame, plot_image_resized))
            
            try:
                self.plot_pub.publish(self.bridge.cv2_to_imgmsg(combined_image, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(e)
        else:
            try:
                self.plot_pub.publish(self.bridge.cv2_to_imgmsg(plot_image, "bgr8"))
            except CvBridgeError as e:
                rospy.logerr(e)

    def calculate_velocities(self):
        for track_id in self.trajectories:
            velocities = []
            # print("Here 1")
            try:
                try:
                    for i in range(0, 3):
                        velocities[i] = np.array(self.trajectories[list(self.trajectories.keys())[0]][-(i + 1)]) - np.array(self.trajectories[list(self.trajectories.keys())[0]][-(i + 2)])

                    self.velocities[track_id] = list(np.mean(velocities))
                    
                except:
                    self.velocities[track_id] = list(np.array(self.trajectories[list(self.trajectories.keys())[0]][-1]) - np.array(self.trajectories[list(self.trajectories.keys())[0]][-2]))
                
                    # print("Here 2")
                    
            except:
                self.velocities[track_id] = [0, 0] # self.trajectories[track_id][-1] - self.trajectories[track_id][-2]
                # print("Here 3")


            # print(self.velocities)

    def calculate_directions(self):
        for track_id in self.trajectories:
            directions = []
            print("Here 1")
            try:
                print("Here 2")
                
                for i in range(1, 2):
                    velocities[i] = list(np.array(self.trajectories[list(self.trajectories.keys())[0]][-i]) - np.array(self.trajectories[list(self.trajectories.keys())[0]][-(i + 1)]))

                self.velocities[track_id] = np.mean(velocities)
            except:
                pass
            #     print("Here 3")
            #     self.velocities[track_id] = 0 # self.trajectories[track_id][-1] - self.trajectories[track_id][-2]


            # print(self.velocities)        


    def publish_trajectories_and_velocities(self):
        # Convert trajectories and velocities to JSON strings
        trajectories_json = json.dumps(self.trajectories)
        velocities_json = json.dumps(self.velocities)

        # Create ROS messages from the JSON strings
        trajectories_msg = String(data=trajectories_json)
        velocities_msg = String(data=velocities_json)

        # Publish the messages
        # self.trajectories_pub.publish(trajectories_msg)
        # self.velocities_pub.publish(velocities_msg)


if __name__ == '__main__':
    try:
        tracker = HumanTrajectoryTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
