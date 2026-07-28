#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /workspaces/limo_ws/install/setup.bash ]; then
  source /workspaces/limo_ws/install/setup.bash
fi

exec "$@"
