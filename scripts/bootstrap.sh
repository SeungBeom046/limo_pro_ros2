#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/limo_ws

if [ ! -d src/limo_ros2 ]; then
  git clone --branch humble https://github.com/agilexrobotics/limo_ros2.git src/limo_ros2
else
  echo "src/limo_ros2 already exists; skipping clone."
fi

rosdep update
rosdep install --from-paths src --ignore-src -r -y

colcon build --symlink-install

echo
echo "Build complete."
echo "Run: source install/setup.bash"
