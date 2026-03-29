#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR
source ./install/setup.bash

cleanup() {
    echo "Shutting down..."
    kill $(jobs -p) 2>/dev/null
    wait $(jobs -p) 2>/dev/null
    pkill -f "Model.x86_64" 2>/dev/null
}
trap cleanup SIGINT SIGTERM

# Start Unity simulation environment
./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &
sleep 3

# Start ROS2 autonomy stack in background
ros2 launch vehicle_simulator system_simulation.launch &
sleep 10

# Start ZMQ bridge node
python3 ./src/utilities/zmq_bridge/zmq_bridge_node.py &

wait
