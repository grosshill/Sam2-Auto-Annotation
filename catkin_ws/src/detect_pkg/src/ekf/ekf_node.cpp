#include <ekf_node.h> // 假设头文件名为这个
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include <deque>
#include <string>
#include <fstream>

class EKFNode {
public:
    EKFNode(const ros::NodeHandle& nh): nh_(nh){
        setup_sub();
        setup_pub();
    }

    void est_and_pub() {
        Eigen::Vector3d vision_odom;
        vision_odom << vision_target.pose.pose.position.x,
                       vision_target.pose.pose.position.y,
                       vision_target.pose.pose.position.z;

        if (!init_flag) {
            estimator->init(vision_odom);
            last_update_time = vision_target.header.stamp.toSec();
            init_flag = true;
            return;
        }

        double curr_time = vision_target.header.stamp.toSec();
        Eigen::Matrix<double, 6, 1> filtered_state = estimator->predictAndUpdate(vision_odom, curr_time);

        nav_msgs::Odometry filtered_odom;
        filtered_odom.header.stamp = vision_target.header.stamp;
        filtered_odom.pose.pose.position.x = filtered_state(0);
        filtered_odom.pose.pose.position.y = filtered_state(1);
        filtered_odom.pose.pose.position.z = filtered_state(2);
        filtered_odom.twist.twist.linear.x = filtered_state(3);
        filtered_odom.twist.twist.linear.y = filtered_state(4);
        filtered_odom.twist.twist.linear.z = filtered_state(5);
        pub.publish(filtered_odom);
    }

private:
    ros::NodeHandle nh_;
    nav_msgs::Odometry vision_target;

    ros::Subscriber sub;
    ros::Publisher pub;
    bool init_flag = false;
    bool new_vision_flag = false;
    double last_update_time = 0;
    // std::vector<std::shared_ptr<KalmanFilter>> estimator;
    std::shared_ptr<KalmanFilter> estimator = std::make_shared<KalmanFilter>();
    void setup_sub() {
        sub = nh_.subscribe<nav_msgs::Odometry>(
            "/ekf_node/input_global_target",
            1,
            [this](const nav_msgs::Odometry::ConstPtr& msg) {
                vision_target = *msg;
                new_vision_flag = true;
                est_and_pub();
            });
    }

    void setup_pub() {
        pub = nh_.advertise<nav_msgs::Odometry>(
            "/ekf_node/target/",
            1);
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "ekf_node");
    ros::NodeHandle nh("~");
    EKFNode ekf_node(nh);
    ros::spin();
    return 0;
}