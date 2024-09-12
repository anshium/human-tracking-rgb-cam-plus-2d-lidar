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
        
        self.message_counter = 0

    def odom_callback(self, data):
        # Initialize message counter if not already initialized
        if self.message_counter is None:
            self.message_counter = 0
        
        # Skip the first N messages
        skip_messages = 200  # Number of messages to skip
        if self.message_counter < skip_messages:
            self.message_counter += 1
        else:
        
            # Extract position from Odometry message
            position = data.pose.pose.position
            self.x_coords.append(position.x)
            self.y_coords.append(position.y)
            
            window_size = 5
            
            # Ensure we have more than one coordinate to compute a heading
            if len(self.x_coords) > window_size:
                # If we have fewer points than the window size, use the available points
                if len(self.x_coords) < window_size:
                    window_size = len(self.x_coords) - 1  # Subtract 1 since we need at least 2 points

                # Compute the deltas for the window size
                dx_sum = 0
                dy_sum = 0
                for i in range(1, window_size + 1 if len(self.x_coords) > window_size else len(self.x_coords)):
                    dx_sum += self.x_coords[-i] - self.x_coords[-(i + 1)]
                    dy_sum += self.y_coords[-i] - self.y_coords[-(i + 1)]

                # Calculate the average dx and dy
                avg_dx = dx_sum / window_size
                avg_dy = dy_sum / window_size

                # Compute the heading using the average deltas
                heading = atan2(avg_dy, avg_dx)
                heading_deg = degrees(heading) / 50
            else:
                heading_deg = 0  # No movement, no heading
            
            self.headings.append(heading_deg)
            
            # Set initial heading if not set
            if self.initial_heading is None:
                self.initial_heading = heading_deg
            
            # Compute cumulative sum of headings
            cumulative_headings = []
            cumulative_sum = 0
            for heading in self.headings:
                cumulative_sum = (cumulative_sum + heading) % 360
                cumulative_headings.append(cumulative_sum)

            self.current_heading = cumulative_headings[-1]

            # Generate and publish the plot
            self.plot_trajectory(cumulative_headings)
        


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
