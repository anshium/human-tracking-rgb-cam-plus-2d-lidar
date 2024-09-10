import cv2
import numpy as np
import pyrealsense2 as rs

# Configure depth stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

try:
    # Wait for a coherent pair of frames: depth
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        print("Could not acquire depth frame")
        exit()

    # Convert depth to numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    # Normalize the depth image to fall between 0 (near) and 1 (far)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Display the result
    cv2.imshow('RealSense', depth_colormap)
    cv2.waitKey(0)

finally:
    # Stop streaming
    pipeline.stop()
