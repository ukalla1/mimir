cmake_minimum_required(VERSION 3.5)
project(realsense_zmq)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

#=====Library zmq_color_image_publisher ======
add_library(zmq_color_image_publisher)
target_sources(zmq_color_image_publisher
  PRIVATE
    src/lib/zmq_color_image_publisher.cpp
)
target_link_libraries(zmq_color_image_publisher zmq)
ament_target_dependencies(zmq_color_image_publisher
  rclcpp
  sensor_msgs
)
target_include_directories(zmq_color_image_publisher PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/realsense_zmq ${CMAKE_CURRENT_SOURCE_DIR}/include/external
)
install(TARGETS zmq_color_image_publisher DESTINATION lib)

#=====Library ros_realsense_subscriber ======
add_library(ros_realsense_subscriber)
target_sources(ros_realsense_subscriber
  PRIVATE
    src/lib/ros_realsense_subscriber.cpp
)
ament_target_dependencies(ros_realsense_subscriber
  rclcpp
)
target_include_directories(ros_realsense_subscriber PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/realsense_zmq ${CMAKE_CURRENT_SOURCE_DIR}/include/external
)
install(TARGETS ros_realsense_subscriber DESTINATION lib)

#=====Executable realsense_zmq_bridge_node ======
add_executable(realsense_zmq_node src/realsense_zmq_node.cpp)
target_link_libraries(realsense_zmq_node ros_realsense_subscriber zmq_color_image_publisher)
ament_target_dependencies(realsense_zmq_node
  rclcpp
  sensor_msgs
)
target_include_directories(realsense_zmq_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/realsense_zmq ${CMAKE_CURRENT_SOURCE_DIR}/include/external
)
install(TARGETS realsense_zmq_node DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
