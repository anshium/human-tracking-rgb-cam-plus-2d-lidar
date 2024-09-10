cmake_minimum_required(VERSION 3.0.2)
project(coordinates)

# Find catkin and any catkin packages

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  laser_geometry
)

find_package(OpenCV REQUIRED)

# Declare a catkin package
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES coordinates
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
include_directories(/usr/include/eigen3)
# Declare a C++ executable
add_executable(coordinates_publisher src/coordinates_publisher.cpp)

# Specify libraries to link a library or executable target against
target_link_libraries(coordinates_publisher
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
