#include "realsense_zmq/zmq_color_image_publisher.hpp"

#include <cstring>
#include <nlohmann/json.hpp>

using nlohmann::json;

ZmqImagePublisher::ZmqImagePublisher(zmq::context_t &ctxRef,
                                     const std::string &endpoint,
                                     const std::string &topicStr)
    : ctx(ctxRef),
      socket(ctx, zmq::socket_type::pub),
      topic(topicStr)
{
    socket.setsockopt(ZMQ_SNDHWM, 1);
    socket.setsockopt(ZMQ_LINGER, 0);
    socket.bind(endpoint);
}

std::string ZmqImagePublisher::buildHeaderJson(
    const sensor_msgs::msg::Image &img,
    const sensor_msgs::msg::CameraInfo *camInfo)
{
    json j;
    j["msgType"] = "image";
    j["width"] = img.width;
    j["height"] = img.height;
    j["encoding"] = img.encoding;
    j["step"] = img.step;
    j["stampSec"] = img.header.stamp.sec;
    j["stampNanosec"] = img.header.stamp.nanosec;
    j["frameId"] = img.header.frame_id;

    if (camInfo != nullptr)
    {
        json jc;
        jc["stampSec"] = camInfo->header.stamp.sec;
        jc["stampNanosec"] = camInfo->header.stamp.nanosec;
        jc["frameId"] = camInfo->header.frame_id;
        jc["width"] = camInfo->width;
        jc["height"] = camInfo->height;
        jc["distortionModel"] = camInfo->distortion_model;
        jc["binningX"] = camInfo->binning_x;
        jc["binningY"] = camInfo->binning_y;
        jc["d"] = camInfo->d; // float64[]
        jc["k"] = camInfo->k; // float64[9]
        jc["r"] = camInfo->r; // float64[9]
        jc["p"] = camInfo->p; // float64[12]
        json roi;
        roi["xOffset"] = camInfo->roi.x_offset;
        roi["yOffset"] = camInfo->roi.y_offset;
        roi["height"] = camInfo->roi.height;
        roi["width"] = camInfo->roi.width;
        roi["doRectify"] = camInfo->roi.do_rectify;
        jc["roi"] = roi;
        j["cameraInfo"] = jc;
    }

    return j.dump(); 
}

void ZmqImagePublisher::sendImage(
    const sensor_msgs::msg::Image &img,
    const sensor_msgs::msg::CameraInfo *camInfo)
{
    std::string headerJson = buildHeaderJson(img, camInfo);

    zmq::message_t topicMsg(topic.size());
    std::memcpy(topicMsg.data(), topic.data(), topic.size());

    zmq::message_t headerMsg(headerJson.size());
    std::memcpy(headerMsg.data(), headerJson.data(), headerJson.size());

    zmq::message_t imageMsg(img.data.size());
    if (!img.data.empty())
    {
        std::memcpy(imageMsg.data(), img.data.data(), img.data.size());
    }

    socket.send(topicMsg, zmq::send_flags::sndmore);
    socket.send(headerMsg, zmq::send_flags::sndmore);
    socket.send(imageMsg, zmq::send_flags::none);
}
