#pragma once

#include <string>
#include <zmq.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

class ZmqImagePublisher
{
public:
    ZmqImagePublisher(zmq::context_t &ctx,
                      const std::string &endpoint,
                      const std::string &topic);

    void sendImage(const sensor_msgs::msg::Image &img,
                   const sensor_msgs::msg::CameraInfo *camInfo);

private:
    zmq::context_t &ctx;
    zmq::socket_t socket;
    std::string topic;

    std::string buildHeaderJson(const sensor_msgs::msg::Image &img,
                                const sensor_msgs::msg::CameraInfo *camInfo);
};
