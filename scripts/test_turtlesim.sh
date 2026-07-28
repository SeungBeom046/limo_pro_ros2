#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/humble/setup.bash
export DISPLAY="${DISPLAY:-host.docker.internal:0}"
export QT_X11_NO_MITSHM=1

ros2 run turtlesim turtlesim_node
