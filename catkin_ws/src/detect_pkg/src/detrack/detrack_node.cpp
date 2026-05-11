#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include "BYTETracker.h"
#include "YoloDet26Api.h"

/*
    Subscribe msg: raw image data from RGB camera.
    Publish msg: tracked object 2D bounding boxes, or may be reconstruct to 3D position.
*/


class DetrackNode {
public:
    explicit DetrackNode(const ros::NodeHandle& nh)
        : nh_(nh), tracker_(30, 30), detector_(YoloDet26_Create()) {
        init();
    }

    ~DetrackNode() {
        if (detector_ != nullptr) {
            YoloDet26_Destroy(detector_);
        }
    }

    void est_and_pub() {
        if (current_frame.empty()) {
            ROS_WARN("Current frame is empty, skipping estimation.");
            return;
        }

        YoloDetResult det_result;
        int ret = YoloDet26_Inference(detector_, current_frame, det_result);
        if (ret != 0) {
            ROS_ERROR("Inference failed with error code: %d", ret);
            return;
        }

        // Convert YoloDetResult to vector<Object> for BYTETracker
        vector<Object> objects;
        for (int i = 0; i < det_result.num; ++i) {
            Object obj;
            obj.rect.x = det_result.boxes[i].left;
            obj.rect.y = det_result.boxes[i].top;
            obj.rect.width = det_result.boxes[i].right - det_result.boxes[i].left;
            obj.rect.height = det_result.boxes[i].bottom - det_result.boxes[i].top;
            obj.label = det_result.classes[i];
            obj.prob = det_result.scores[i];
            objects.push_back(obj);
        }

        vector<STrack> tracked_objects = tracker_.update(objects);

        // Publish tracked objects as nav_msgs::Odometry
        for (const auto& track : tracked_objects) {
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time::now();
            odom_msg.pose.pose.position.x = track.tlwh[0] + track.tlwh[2] / 2.0; // center x
            odom_msg.pose.pose.position.y = track.tlwh[1] + track.tlwh[3] / 2.0; // center y
            odom_msg.pose.pose.position.z = 0; // Assuming 2D tracking
            pub.publish(odom_msg);
        }
    }

private:
    ros::NodeHandle nh_;
    BYTETracker tracker_;
    YoloDet26Handle detector_;
    ros::Subscriber sub;
    ros::Publisher pub;
    cv::Mat current_frame;
    const float target_size = 0.25f;
    
    void init() {
        if (detector_ != nullptr) {
            YoloDet26_SetModel(detector_, "stage2_green_bg_yellow_target.engine", YOLODET26_BACKEND_GPU);
        }
    }

    void setup_subs() {
        sub = nh_.subscribe(
            "subscribe img msg",
            1,
            [this](const sensor_msgs::Image::ConstPtr& msg) {
                cv_bridge::CvImagePtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                } catch (cv_bridge::Exception& e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                    return;
                }
                /* process_image(cv_ptr->image); */
                current_frame = cv_ptr->image;
                est_and_pub();
            }
        )
    }

    void setup_pubs() {
        pub = nh_.advertise<nav_msgs::Odometry>("publish tracked object msg", 1);
    }

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "bytetrack_node");
    ros::NodeHandle nh;
    DetrackNode node(nh);
    ros::spin();
    return 0;
}