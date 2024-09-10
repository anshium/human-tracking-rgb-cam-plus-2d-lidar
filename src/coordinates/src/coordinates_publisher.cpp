#include "ros/ros.h"
#include "std_msgs/String.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/LaserScan.h"
#include <laser_geometry/laser_geometry.h>
#include <cmath>

using namespace cv;

ros::Publisher pub;

cv::Mat K;
cv::Mat D;
cv::Mat Rcl;
cv::Mat tcl;
laser_geometry::LaserProjection projector_;

void callback(const sensor_msgs::LaserScan::ConstPtr &scan_msg)
{
    double fx = 386.458;
    double fy = 386.458;
    double cx = 321.111;
    double cy = 241.595;

    double k1 = -0.054603107273578644;
    double k2 = 0.06334752589464188;
    double p1 = 0.00022518340847454965;
    double p2 = 0.0002921034465543926;

    K = (Mat_<double>(3, 3) << fx, 0., cx, 0., fy, cy, 0., 0., 1.);
    D = (Mat_<double>(5,1) << k1, k2, p1, p2, 0.0);

    Rcl = (Mat_<double>(3,3) <<   0.998605,    0.0528031, -0.000539675,
                                  -0.0011676,    0.0118618,    -0.999929,
                                   -0.0527929,     0.998534,    0.0119069);
                                
    tcl = (Mat_<double>(3, 1) << -0.231367,  0.250736, 0.0676897);

    // Project the LaserScan to a PointCloud
    sensor_msgs::PointCloud cloud;
    projector_.projectLaser(*scan_msg, cloud);

    std::vector<float> ranges = scan_msg->ranges;
    std::vector<float> angles;
    float angle_increment = scan_msg->angle_increment;

    size_t i = 0;

    for (auto &point : cloud.points)
    {
        // Transform to camera frame
        Mat point_l = (Mat_<double>(3, 1) << point.x, point.y, point.z);
        Mat point_c = Rcl * point_l + tcl;

        if (point_c.at<double>(2, 0) <= 0.)
            continue;

        Mat uv = K * point_c;
        uv /= uv.at<double>(2, 0); 

        double u = uv.at<double>(0, 0);
        double v = uv.at<double>(1, 0);

        float angle = scan_msg->angle_min + i * angle_increment;
        float x = ranges[i] * cos(angle);
        float y = ranges[i] * sin(angle);

        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(x) || !std::isfinite(y)) {
            i++;
            continue;
        }

        std::stringstream ss;
        ss << u << "," << v << "," << x << "," << y;
        // image coordinates, world coordinates
        std_msgs::String string_msg;
        string_msg.data = ss.str();
        pub.publish(string_msg);

        i++;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pub_node");
    ros::NodeHandle nh;

    pub = nh.advertise<std_msgs::String>("/Reprojection_pts", 10);
    ros::Subscriber scan_sub = nh.subscribe<sensor_msgs::LaserScan>("/scan_raw", 10, callback);

    ros::spin();

    return 0;
}