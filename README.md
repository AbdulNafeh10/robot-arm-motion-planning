# Robot Arm Motion Planning

Python motion planning for a UR5 robot in CoppeliaSim. The robot plans around obstacles, moves toward a green target, and follows the same path back.

[![UR5 simulation](media/ur5_simulation.png)](https://youtu.be/wQspVbpZ7rY)

[Watch the robot move](https://youtu.be/wQspVbpZ7rY) | [Watch RRT planning](https://youtu.be/gnm1zqYfLEw)

## What it does

- Uses forward kinematics and inverse kinematics to work with the robot's position.
- Checks joint configurations and path segments for collisions.
- Uses bidirectional RRT-Connect when the direct joint-space path is blocked.
- Moves through the planned waypoints with smooth cubic interpolation.

## Run it

Requires Python 3.11+ and CoppeliaSim with the ZeroMQ remote API enabled. The example was tested in CoppeliaSim 4.10 EDU on Windows.

```powershell
py -m pip install -r requirements.txt
py -m pytest -q
```

Open `scenes/UR5ObstacleDemo.ttt` in CoppeliaSim and **leave the simulation stopped**. In another terminal, from the repository root:

```powershell
py -m src.touch_goal check
py -m src.touch_goal plan
py -m src.touch_goal play
```

`check` looks for a reachable goal pose. `plan` searches for a route and saves it in `output/touch_goal_path.json`. `play` checks the saved route against the current scene before moving the robot. If the scene changes, run `plan` again. Planning may fail if the obstacles leave no valid route.

To plot the planned joint angles after planning:

```powershell
py -m src.plot_touch_path
```

The plot is saved as `media/ur5_joint_trajectory.png`. It shows planned interpolation, not measured joint motion.

## Other examples

- `py -m src.demo_rrt` produces a standalone 2D RRT illustration.
- Open `scenes/Q3Scene.ttt` and run `py -m src.check_scene` to check the original coursework start and goal configurations.

## Notes

This is an offline planner. The robot follows a saved route, not a path continuously replanned during movement. Playback is a kinematic preview with the simulation stopped. Collision checks sample the motion and depend on the scene's collision geometry; contact forces are not simulated. The green sphere is treated as the intended target rather than an obstacle.

The project started as university robotics coursework and was later recovered, tested, and extended with the custom obstacle scene and goal-reaching demo. The separate conceptual system designs from the coursework are not implemented here.
