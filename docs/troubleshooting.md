## Troubleshooting Guides
1. **Colcon build** might explode the ram so the PC freezes
- Solutions: **export MAKEFLAGS="-j 1" colcon build --executor sequential**, this command will use one core and one package to build the workspace

2. Ros2 extension may not be able to work when the simulator starts
- Solutions: **make sure to configure the ROS for Isaac sim**, basically use the fastdds.xml

3. The configuration may still not work, i.e. ros2 command pops up an error saying some configuration errors even after completing step 2
- Solutions: **open up one terminal that set the environment first (export FASTRTPS_DEFAULT_PROFILES_FILE=...), then start the isaac sim.** No idea why it's not working even I put that in the *extra arg* of isaac sim 
4. Docker:
- Docker image build
    - docker build --build-arg GITHUB_TOKEN=$GITHUB_TOKEN  -t my_isaac_ros_image .


- Docker container command
    - docker run --name my_isaac_ros_container \
        --runtime=nvidia --gpus all \
        -e "ACCEPT_EULA=Y" \
        -e "PRIVACY_CONSENT=Y" \
        --network=host \
        -v /home/chris/Chris/placement_ws/src/data:/home/chris/Chris/placement_ws/src/data:rw \
        -it --entrypoint bash my_isaac_ros_image

- Docker for grasping
    - docker build -t my_ros_noetic_image .
    - docker run -it   -v /home/chris/Chris/placement_ws/src/placement_quality/docker_files/ros_ws/src/grasp_generation:/home/ros_ws/src/grasp_generation:rw   my_ros_noetic_image


5. If Vscode does not recognize some certain packages
- Solutions:
    - import the package, then print(package.\_\_file__)

6. Cannot Plot in the docker for gpd:
- Solutions:
  - xhost +local:
  - docker run --gpus all -it \
        -v /home/chris/Chris/placement_ws/src/placement_quality/docker_files:/home \
        --name ros1 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        my-ros1-x11:latest

7. Cluster usage:
- Connections:
    - ssh tianyuanl@gandalf-dev.it.deakin.edu.au

8. Ycb box dimensions:
- use code
  ```
  prim = stage.GetPrimAtPath("/World/_09_gelatin_box")
  bbox = UsdGeom.Boundable(prim).ComputeWorldBound(0.0, UsdGeom.Tokens.default_)
  extent = bbox.GetRange()
  print("Object size in meters:", extent.GetSize())
  ```
- The dimensions are: (0.0891, 0.0731, 0.0299)