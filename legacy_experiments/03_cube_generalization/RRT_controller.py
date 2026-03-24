import os
from typing import Optional

import carb
import numpy as np
import omni.isaac.core.objects
import omni.isaac.motion_generation.interface_config_loader as interface_config_loader
from omni.isaac.franka import Franka
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.motion_generation import ArticulationTrajectory
from omni.isaac.motion_generation.lula import RRT
from omni.isaac.motion_generation.lula.trajectory_generator import LulaCSpaceTrajectoryGenerator
from omni.isaac.motion_generation.path_planner_visualizer import PathPlannerVisualizer
from omni.isaac.motion_generation.path_planning_interface import PathPlanner


class RRTController(BaseController):
    def __init__(
            self,
            name: str,
            robot_articulation: Articulation,
            ground_plane: omni.isaac.core.objects,
            pedestal_planner_box: omni.isaac.core.objects,
            physics_dt: float = 1 / 60.0,
            rrt_interpolation_max_dist: float = 0.01,
        ):
        BaseController.__init__(self, name)

        self.ground_plane = ground_plane
        self.pedestal_planner_box = pedestal_planner_box
        # Load default RRT config files stored in the omni.isaac.motion_generation extension
        rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        rrt_config["end_effector_frame_name"] = "panda_hand"
        rrt = RRT(**rrt_config)

        # Create a trajectory generator to convert RRT cspace waypoints to trajectories
        self._cspace_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            rrt_config["robot_description_path"], rrt_config["urdf_path"]
        )

        # It is important that the Robot Description File includes optional Jerk and Acceleration limits so that the generated trajectory
        # can be followed closely by the simulated robot Articulation
        for i in range(len(rrt.get_active_joints())):
            assert self._cspace_trajectory_generator._lula_kinematics.has_c_space_acceleration_limit(i)
            assert self._cspace_trajectory_generator._lula_kinematics.has_c_space_jerk_limit(i)
        
        
        # Create a visualizer to visualize the path planner
        self._path_planner_visualizer = PathPlannerVisualizer(robot_articulation, rrt)
        self._path_planner: RRT = self._path_planner_visualizer.get_path_planner()
        # self._path_planner.add_obstacle(self.ground_plane, static=True)
        # self._path_planner.add_obstacle(self.pedestal_planner_box, static=True)

        self._robot: Franka = self._path_planner_visualizer.get_robot_articulation() 

        self._last_solution = None
        self._action_sequence = None

        self._physics_dt = physics_dt
        self._rrt_interpolation_max_dist = rrt_interpolation_max_dist
        self.ik_check = None


    def _convert_rrt_plan_to_trajectory(self, rrt_plan):
        interpolated_path = self._path_planner_visualizer.interpolate_path(rrt_plan, self._rrt_interpolation_max_dist)
        trajectory = self._cspace_trajectory_generator.compute_c_space_trajectory(interpolated_path)
        art_trajectory = ArticulationTrajectory(self._robot, trajectory, self._physics_dt)

        return art_trajectory.get_action_sequence()

    def _make_new_plan(
        self, target_end_effector_position: np.ndarray, target_end_effector_orientation: Optional[np.ndarray] = None
    ):
        self._path_planner.set_end_effector_target(target_end_effector_position, target_end_effector_orientation)
        self._path_planner.update_world()

        # In the original script, they created a new PathPlannerVisualizer object here, not sure why
        # path_planner_visualizer = PathPlannerVisualizer(self._robot, self._path_planner)

        active_joints = self._path_planner_visualizer.get_active_joints_subset()
        if self._last_solution is None:
            start_pos = active_joints.get_joint_positions()
        else:
            start_pos = self._last_solution

        self._path_planner.set_max_iterations(5000)
        self._rrt_plan = self._path_planner.compute_path(start_pos, np.array([]))

        if self._rrt_plan is None or len(self._rrt_plan) <= 1:
            carb.log_warn("No plan could be generated to target pose: " + str(target_end_effector_position))
            self._action_sequence = []
            return False


        self._action_sequence = self._convert_rrt_plan_to_trajectory(self._rrt_plan)
        self._last_solution = self._action_sequence[-1].joint_positions
        return True
    def forward(
        self, 
        target_end_effector_position: np.ndarray, 
        target_end_effector_orientation: Optional[np.ndarray] = None
    ) -> ArticulationAction:
        if self._action_sequence is None:
            # This will only happen the first time the forward function is used
            self.ik_check = self._make_new_plan(target_end_effector_position, target_end_effector_orientation)

        if len(self._action_sequence) == 0:
            # The plan is completed; return null action to remain in place
            return ArticulationAction()

        if len(self._action_sequence) == 1:
            final_positions = self._action_sequence[0].joint_positions
            return ArticulationAction(
                final_positions, np.zeros_like(final_positions), joint_indices=self._action_sequence[0].joint_indices
            )

        return self._action_sequence.pop(0)

    def add_obstacle(self, obstacle: omni.isaac.core.objects, static: bool = False) -> None:
        self._path_planner.add_obstacle(obstacle, static)

    def remove_obstacle(self, obstacle: omni.isaac.core.objects) -> None:
        self._path_planner.remove_obstacle(obstacle)

    def reset(self) -> None:
        # PathPlannerController will make one plan per reset
        self._path_planner.reset()
        # self.add_obstacle(self.ground_plane, static=True)
        # self.add_obstacle(self.pedestal_planner_box, static=True)
        self._action_sequence = None
        self._last_solution = None

    def get_path_planner(self) -> PathPlanner:
        return self._path_planner
    

    def is_done(self) -> bool:
        if len(self._action_sequence) <= 1:
            return True
        else:
            return False
