#include "ros_realsense_subscriber.hpp"

using std::placeholders::_1;

RealsenseSubscriber::RealsenseSubscriber(
    const rclcpp::Node::SharedPtr &nodePtr,
    const std::string &imageTopic,
    const std::string &cameraInfoTopic,
    ImageCallback cb)
    : node(nodePtr),
      imageCallback(cb)
{
    imageSub = node->create_subscription<sensor_msgs::msg::Image>(
        imageTopic,
        10,
        std::bind(&RealsenseSubscriber::handleImage, this, _1));

    cameraInfoSub = node->create_subscription<sensor_msgs::msg::CameraInfo>(
        cameraInfoTopic,
        10,
        std::bind(&RealsenseSubscriber::handleCameraInfo, this, _1));
}

void RealsenseSubscriber::handleImage(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    const sensor_msgs::msg::CameraInfo *camPtr = nullptr;
    if (latestCameraInfo)
    {
        camPtr = latestCameraInfo.get();
    }
    if (imageCallback)
    {
        imageCallback(*msg, camPtr);
    }
}

void RealsenseSubscriber::handleCameraInfo(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    latestCameraInfo = msg;
}
