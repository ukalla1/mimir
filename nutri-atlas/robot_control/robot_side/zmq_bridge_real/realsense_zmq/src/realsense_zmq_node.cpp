#include <rclcpp/rclcpp.hpp>
#include <zmq.hpp>

#include "zmq_color_image_publisher.hpp"
#include "ros_realsense_subscriber.hpp"
#include "zmq.ports.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("realsense_zmq_bridge_node");

    std::string colorImageTopic =
        node->declare_parameter<std::string>("color_image_topic",
                                             "/camera/color/image_raw");
    std::string colorCameraInfoTopic =
        node->declare_parameter<std::string>("color_camera_info_topic",
                                             "/camera/color/camera_info");
    std::string colorZmqEndpoint =
        node->declare_parameter<std::string>("color_zmq_endpoint",
                                             PORT_COLOR_IMAGE);
    std::string colorZmqTopic =
        node->declare_parameter<std::string>("color_zmq_topic",
                                             "realsense/color");

    std::string depthImageTopic =
        node->declare_parameter<std::string>("depth_image_topic",
                                             "/camera/depth/image_rect_raw");
    std::string depthCameraInfoTopic =
        node->declare_parameter<std::string>("depth_camera_info_topic",
                                             "/camera/depth/camera_info");
    std::string depthZmqEndpoint =
        node->declare_parameter<std::string>("depth_zmq_endpoint",
                                             PORT_DEPTH_IMAGE);
    std::string depthZmqTopic =
        node->declare_parameter<std::string>("depth_zmq_topic",
                                             "realsense/depth");

    zmq::context_t ctx(1);
    auto colorZmqPub =
        std::make_shared<ZmqImagePublisher>(ctx, colorZmqEndpoint, colorZmqTopic);

    RealsenseSubscriber colorSub(
        node,
        colorImageTopic,
        colorCameraInfoTopic,
        [colorZmqPub](const sensor_msgs::msg::Image &img,
                      const sensor_msgs::msg::CameraInfo *camInfo)
        {
            colorZmqPub->sendImage(img, camInfo);
        });

    auto depthZmqPub =
        std::make_shared<ZmqImagePublisher>(ctx, depthZmqEndpoint, depthZmqTopic);

    RealsenseSubscriber depthSub(
        node,
        depthImageTopic,
        depthCameraInfoTopic,
        [depthZmqPub](const sensor_msgs::msg::Image &img,
                      const sensor_msgs::msg::CameraInfo *camInfo)
        {
            depthZmqPub->sendImage(img, camInfo);
        });

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
