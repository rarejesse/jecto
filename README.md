# JECTO
## Generate and visualize 3D motion, simulate sensor data, and support the development visual-inertial navigation systems.

Specify a visual-inertial scene containing a set of related trajectories, a sensor platform, and features, in a single configuration yaml file.

See the [parameter reference](parameter_reference.md).
<details>
<summary><b>example_config_01.yaml</b>: bspline trajectory, 2 cameras, 1 imu, point features</summary>

```yaml
trajectory_group:
    reference: generated
    trajectories:
      - id: generated
        translation_trajectory:
          bspline:
            res: 50
            degree: 5
            span_time: 1
            control_points:
              points:
                - [ 10.00,  0.00,  0.00]
                - [ 10.00,  0.00,  0.00]
                - [ 10.00,  0.00,  0.00]
                - [ 10.00,  0.00,  0.00]
                - [ 9.01,  2.34,  1.00]
                - [ 6.24,  7.82,  1.00]
                - [ 2.23,  2.75,  2.00]
                - [ 2.23,  9.75,  1.00]
                - [-6.24,  7.82,  0.00]
                - [-9.01,  4.34,  4.20]
                - [-10.00,  0.00,  3.00]
                - [-10.01,  0.74, -2.00]
                - [-8.23, -7.82,  0.00]
                - [-2.23, -9.75,  6.00]
                - [ 2.23, -9.75,  3.40]
                - [ 6.24,  1.82,  0.00]
                - [ 9.01, -4.34,  0.00]
                - [ 9.50, -0.00,  0.00]
                - [ 10.00, -0.00,  0.00]
                - [ 10.00, -0.00,  0.00]
                - [ 10.00, -0.00,  0.00]
                - [ 10.00, -0.00,  0.00]
        rotation_trajectory:
          align_axis:
            axis: z
            vec:
              current_trajectory:  centroid
              negate: false
            grounded_axis: y
            flip: true

platform:
  id: 'stereo'
  base_frame: 'imu0'
  sensors:
    - id: 'imu0'
      enable_measurements: true
      rate: 100
      transform:
        rotation: [0,0,0]
        translation: [0,0,0]
      imu:
        gyro_noise_density: 0.001
        gyro_random_walk: 0.0001
        accel_noise_density: 0.01
        accel_random_walk: 0.001
      time_offset: 0.0
    - id: 'cam0'
      enable_measurements: true
      rate: 25
      transform:
        from: 'imu0'
        translation: [-0.3,0.2,0.3]
        rotation: [0,0,-90]
      camera:
        height: 480
        width: 640
        intrinsics: [300,300,325,242]
        distortion: [-0.01, 0.01, 0.00019359, 1.76187114e-05]
      time_offset: -0.05

    - id: 'cam1'
      enable_measurements: true
      rate: 25
      transform:
        from: 'cam0'
        translation: [-0.4,0.0,0.0]
        rotation: [0,0,0]
      camera:
        height: 480
        width: 640
        intrinsics: [300,300,325,242]
      time_offset: -0.05

  body_frames:
    - id: test_frame
      transform:
        rotation: [30,15,0]
        translation: [1,2,0]

features:
  - id: feats0
    color: white
    points:
      - [6.2,0.1,0.5]
      - [6.1,0.3,0.4]
      - [5.5,0.3,0.4]

  - id: feats1
    random_points:
      num: 100
      center: [-2,0,2]
      radius: 2

  - id: feats2
    color: green
    planar_points:
      num: 50
      center: [2,2,0]
      radius: 2
      normal: [0, 0, -1]

  - id: feats3
    color: cyan
    planar_points:
      center: [0,8,2]
      normal: [0, -1, -1]
      radius: 4
      grid_spacing: 0.5

  - id: feats4
    color: purple
    planar_points:
      num: 50
      center: [0,-5,2]
      normal: [3, -1, 1]
      radius: 2

  - id: feats5
    color: yellow
    planar_points:
      center: [-2,0,8]
      normal: [0,0,1]
      radius: 8
      grid_spacing: 0.5
```
</details>



https://github.com/jessebloecker/vinlab/assets/29802265/985267c3-2f20-4b27-b573-2a3f9129169e




Vinlab contains Python and C++ and uses [Scipy](https://scipy.org/), [OpenCV](https://opencv.org/), [ROS2](https://docs.ros.org/en/humble/index.html), and [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html)


## Install
todo: put instructions here

## Run Demo

run the scene viewer
~~~
ros2 run vinlab scene_viewer.py --ros-args --params-file /path/to/ros2_ws/src/vinlab/config/scene_viewer.yaml
~~~

run the camera simulator
~~~
ros2 run vinlab cam_simulator_node
~~~

run rviz:
~~~
rviz2 -d ~/path/to/config.rviz'
~~~




