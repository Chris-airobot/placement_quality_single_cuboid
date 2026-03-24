# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import typing
# from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
import numpy as np
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper
from scipy.spatial.transform import Rotation, Slerp
from helper import get_current_end_effector_pose, draw_frame

def convert_wxyz_to_xyzw(q_wxyz):
    """Convert a quaternion from [w, x, y, z] format to [x, y, z, w] format."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]

class MyBaseController(BaseController):
    """
    A simple pick and place state machine for tutorials

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

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
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False

        # Set a position threshold for early stage completion
        self._position_threshold = 0.01
        # Store the current target position for reference
        self._current_ee_target_position = None
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        grasping_orientation: typing.Optional[np.ndarray] = None,
        placement_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        end_effector_offset = np.array([0, 0, 0]) if end_effector_offset is None else end_effector_offset
        grasping_orientation = np.array([0.0, np.pi, 0.0]) if grasping_orientation is None else grasping_orientation
        placement_orientation = np.array([0.0, np.pi, 0.0]) if placement_orientation is None else placement_orientation

        # if self._current_ee_target_position is not None:        
        #     # print(f"now you are pausing")    
        #     self._is_stage_task_done()
        #     self.pause()


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
            ### Position Section
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
            ### Orientation Section
            if len(grasping_orientation) == 3:
                grasping_orientation = euler_angles_to_quat(grasping_orientation)
            if len(placement_orientation) == 3:
                placement_orientation = euler_angles_to_quat(placement_orientation)
            interpolated_orientation = self._get_slerp_quat(grasping_orientation, placement_orientation)

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, 
                target_end_effector_orientation=interpolated_orientation
            )

        self._t += self._events_dt[self._event]
        # print(f"Event: {self._event}, Your stage time limit: {stage_time_limit}")
        # Case for stages other than 2, 3, 7
        if stage_time_limit == None:
            # Define a maximum allowed time for the stage (e.g., 2 seconds)
            stage_time_limit = 2.0
            if self._t >= stage_time_limit or self.is_stage_task_done(self._current_ee_target_position, self._position_threshold):

                self._event += 1
                self._t = 0
        else:
            if self._t >= stage_time_limit:
                self._event += 1
                self._t = 0

        return target_joint_positions


    
    def is_stage_task_done(self, ee_target_position, threshold) -> bool:
        """
        Checks if the end-effector has reached the target position within a tolerance.
        Replace the _get_current_end_effector_position() method with actual sensor feedback.
        """
        current_position, current_orientation = get_current_end_effector_pose()

        if ee_target_position is not None:            
            # draw_frame(current_position, current_orientation)
            pos_error = np.linalg.norm(current_position - ee_target_position)
            # print(f"current_position: {current_position}")
            # print(f"current_ee_target_position: {self._current_ee_target_position}")
            # print(f"pos_error: {pos_error}")
            # Print error for debugging (optional)
            return (pos_error < threshold)
        return False

    


    def _get_slerp_quat(self, q_start, q_end):
        """
        Interpolates between two quaternions using SciPy's Slerp.
        
        Args:
            q_start (array-like): Starting quaternion [x, y, z, w].
            q_end (array-like): Target quaternion [x, y, z, w].
            alpha (float): Interpolation factor between 0 (start) and 1 (target).

        Returns:
            np.ndarray: Interpolated quaternion [x, y, z, w].
        """
        
        alpha = self._get_alpha()
        key_times = [0, 1]
        key_rots = Rotation.from_quat([convert_wxyz_to_xyzw(q_start), convert_wxyz_to_xyzw(q_end)])
                                       
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([alpha])


        return np.array([interp_rot.as_quat()[0][3], 
                         interp_rot.as_quat()[0][0], 
                         interp_rot.as_quat()[0][1], 
                         interp_rot.as_quat()[0][2]])


    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

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
            raise ValueError()

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
            raise ValueError()
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
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return
