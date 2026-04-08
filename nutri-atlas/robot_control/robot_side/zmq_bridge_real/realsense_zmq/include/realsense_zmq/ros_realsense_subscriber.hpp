#pragma once

#include <functional>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class RealsenseSubscriber
{
public:
    using ImageCallback =
        std::function<void(const sensor_msgs::msg::Image &,
                           const sensor_msgs::msg::CameraInfo *)>;

    RealsenseSubscriber(const rclcpp::Node::SharedPtr &node,
                        const std::string &imageTopic,
                        const std::string &cameraInfoTopic,
                        ImageCallback cb);

private:
    void handleImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void handleCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    rclcpp::Node::SharedPtr node;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfoSub;

    sensor_msgs::msg::CameraInfo::SharedPtr latestCameraInfo;
    ImageCallback imageCallback;
};
