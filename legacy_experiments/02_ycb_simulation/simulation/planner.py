# CombinedController.py

import typing
import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.manipulators.grippers.gripper import Gripper
from scipy.spatial.transform import Rotation, Slerp
from utils.helper import get_current_end_effector_pose,convert_wxyz_to_xyzw



class YcbPlanner(BaseController):
    """
    Combined controller that integrates the low-level articulation controller 
    (i.e. the robot's movement execution using an RMPFlowController) and the 
    high-level motion planning/state machine.

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.


    Args:
        name (str): Identifier for the controller.
        gripper (ParallelGripper): Gripper controller for open/close actions.
        robot_articulation (Articulation): The robot's articulation object.
        end_effector_initial_height (Optional[float], optional): 
            Initial height for the end effector. Defaults to None.
        events_dt (Optional[List[float]], optional): 
            List of time steps for each state machine event. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        # Use DataCollectionController defaults if none provided.
        if events_dt is None:
            events_dt = [0.008, 0.005, 1, 0.1, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height if end_effector_initial_height is not None else 0.3 / get_stage_units()
        self._h0 = None

        # Ensure events_dt is a list of correct length.
        self._events_dt = events_dt
        if not isinstance(self._events_dt, (np.ndarray, list)):
            raise Exception("events dt need to be list or numpy array")
        if isinstance(self._events_dt, np.ndarray):
            self._events_dt = self._events_dt.tolist()
        if len(self._events_dt) > 10:
            raise Exception("events dt length must be less than 10")

        # Instantiate the motion planner (cspace controller) using RMPFlowController.
        self._cspace_controller = RMPFlowController(name=name + "_cspace_controller", robot_articulation=robot_articulation)
        self._gripper = gripper
        self._pause = False

        # Set a position threshold for early stage completion.
        self._position_threshold = 0.01
        # Store the current target position for reference.
        self._current_ee_target_position = None

    def is_paused(self) -> bool:
        return self._pause

    def get_current_event(self) -> int:
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        picking_orientation: typing.Optional[np.ndarray] = None,
        placement_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Runs one step of the controller state machine.

        Depending on the event (phase), this method computes the target joint positions.
        """
        end_effector_offset = np.array([0, 0, 0]) if end_effector_offset is None else end_effector_offset
        picking_orientation = np.array([0.0, np.pi, 0.0]) if picking_orientation is None else picking_orientation
        placement_orientation = np.array([0.0, np.pi, 0.0]) if placement_orientation is None else placement_orientation

        stage_time_limit = None
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event == 2:
            stage_time_limit = 1.0
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            stage_time_limit = 1.0
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            stage_time_limit = 1.0
            target_joint_positions = self._gripper.forward(action="open")
        else:
            # Update the current target for initial phases.
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(placing_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )

            if self._event in [0, 4, 5, 8, 9]:
                self._current_ee_target_position = position_target
            elif self._event == 1:
                self._current_ee_target_position = picking_position
            elif self._event == 6:
                self._current_ee_target_position = placing_position

            # Convert Euler angles to quaternions if necessary.
            if len(picking_orientation) == 3:
                picking_orientation = euler_angles_to_quat(picking_orientation)
            if len(placement_orientation) == 3:
                placement_orientation = euler_angles_to_quat(placement_orientation)
            interpolated_orientation = self._get_slerp_quat(picking_orientation, placement_orientation)

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, 
                target_end_effector_orientation=interpolated_orientation
            )

        self._t += self._events_dt[self._event]
        if stage_time_limit is None:
            stage_time_limit = 1.5
            if self._t >= stage_time_limit or self.is_stage_task_done(self._current_ee_target_position, self._position_threshold):
            # if self._t >= stage_time_limit:
                self._event += 1
                self._t = 0
        else:
            if self._t >= stage_time_limit:
                self._event += 1
                self._t = 0

        return target_joint_positions


    def is_stage_task_done(self, ee_target_position, threshold) -> bool:
        """
        Checks if the end-effector has reached the target position within a specified tolerance.
        """
        current_position, current_orientation = get_current_end_effector_pose()
        if ee_target_position is not None:
            # print(f"position error: {np.linalg.norm(current_position - ee_target_position)}")
            pos_error = np.linalg.norm(current_position - ee_target_position)
            return pos_error < threshold
        return False
    

    def _get_slerp_quat(self, q_start, q_end):
        """
        Interpolates between two quaternions using SciPy's Slerp.
        """
        alpha = self._get_alpha()
        key_times = [0, 1]
        key_rots = Rotation.from_quat([convert_wxyz_to_xyzw(q_start), convert_wxyz_to_xyzw(q_end)])
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([alpha])
        quat = interp_rot.as_quat()[0]
        # Convert back from [x, y, z, w] to [w, x, y, z]
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        return (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError("Invalid event state")

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError("Invalid event state")
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events dt need to be list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False
