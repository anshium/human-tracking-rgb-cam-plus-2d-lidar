#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from math import atan2, degrees

class WheelchairTrajectoryTracker:
    def __init__(self):
        rospy.init_node('wheelchair_trajectory_tracker', anonymous=True)

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.plot_pub = rospy.Publisher("/trajectory_plot_2", Image, queue_size=10)
        self.bridge = CvBridge()
        
        # Initialize lists to store x, y coordinates and headings
        self.x_coords = []
        self.y_coords = []
        self.headings = []
        
        # Initialize the initial heading
        self.initial_heading = None
        self.current_heading = None

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

        # Generate and publish the plot
        self.plot_trajectory(relative_headings)

    def plot_trajectory(self, relative_headings):
        # Create a new figure
        plt.figure(figsize=(12, 6))
        
        # Plot trajectory
        plt.subplot(1, 2, 1)
        plt.plot(self.x_coords, self.y_coords, marker='o', linestyle='-', color='b')
        plt.title('Wheelchair Trajectory')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Plot heading change
        plt.subplot(1, 2, 2)
        plt.plot(range(len(relative_headings)), relative_headings, marker='x', linestyle='-', color='r')
        plt.title('Relative Heading Change')
        plt.xlabel('Step')
        plt.ylabel('Heading Change (degrees)')
        plt.grid(True)
        
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Convert BytesIO object to a numpy array
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        
        # Decode the numpy array into an image
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            rospy.logerr("Failed to decode image.")
            return
        
        # Convert the image to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                
        # Publish the ROS Image message
        self.plot_pub.publish(img_msg)


if __name__ == '__main__':
    tracker = WheelchairTrajectoryTracker()
    rospy.spin()
