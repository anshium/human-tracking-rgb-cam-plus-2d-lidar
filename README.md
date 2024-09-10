Initially created by @schlechter-afk, @anshium (me) and @samkit-2512j.

Highly modified later

### Running Instruction

1. Remove devel and build dirs (I mistakenly pushed them - and lazy to remove)
2. `catkin_make`
3. `source devel/setup.bash`
4. `rosrun det_and_track combined_script.py`

### Topics needed
1. Camera Color frame 
2. 2D LIDAR Scan
3. Odometry
4. (Optional) Camera depth frame.

### Topics Published (Not all published by default)
1. `/trajectory_plot`
2. `/point_cloud`
3. `/trajectories`
4. `/velocities`
5. `/distances` (Not implemented yet)
