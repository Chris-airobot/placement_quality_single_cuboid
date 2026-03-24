from isaacsim import SimulationApp
CONFIG = {"headless": False}
simulation_app = SimulationApp(CONFIG)

# import carb

import numpy as np
import asyncio

from omni.isaac.core import World
from omni.isaac.core.utils import extensions
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from utils.camera_utility import *
# from models.process_data_helpers import *
# from models.dataset import *
# from models.model_use import single_model_test, predict_single_sample
# from omni.isaac.core.utils.rotations import  quat_to_euler_angles
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()



# def encode_outputs(outputs: dict):
#     """
#     - Head A (classification): feasibility = 0 or 1
#         If `grasp_unsuccessful` OR `bad` is True => 0 (fail), else 1 (success).
#     - Head B (regression): [pose_diff, ori_diff, shift_pos, shift_ori, contacts]
#     """
#     # Classification (feasibility)
#     is_fail = (outputs.get("grasp_unsuccessful", False) or 
#                 outputs.get("bad", False))
#     feasibility_label = 0 if is_fail else 1  # 0=fail, 1=success

#     pos_diff = outputs.get("position_difference", None)
#     ori_diff = outputs.get("orientation_difference", None)
#     shift_pos = outputs.get("shift_position", None)
#     shift_ori = outputs.get("shift_orientation", None)
#     contacts = outputs.get("contacts", None)

#     # Convert Nones to 0.0 or some default
#     # (Alternatively, you could skip these samples)
#     if pos_diff is None: 
#         pos_diff = 0.0
#     if ori_diff is None:
#         ori_diff = 0.0
#     if shift_pos is None:
#         shift_pos = 0.0
#     if shift_ori is None:
#         shift_ori = 0.0
#     if contacts is None:
#         contacts = 0


#     # Make them floats
#     pos_diff  = float(pos_diff)
#     ori_diff  = float(ori_diff)
#     shift_pos = float(shift_pos)
#     shift_ori = float(shift_ori)
#     contacts  = float(contacts)

#     # "values are:pose_diffs: 2.0033138697629886, ori_diffs: 2.9932579711083727, shift_poss: 0.13525934849764623, shift_oris: 1.6673673523277988, contacts: 5.0"
#     pos_diff_max = 0.6081497916935493
#     ori_diff_max = 2.848595486712802
#     shift_pos_max = 0.0794796857415393
#     shift_ori_max = 2.1095218360699306
#     contacts_max = 4.0

#     params = {
#         'h_diff_weight': 0.6,
#         'pos_weight': 0.3,
#         'h_shift_weight': 0.2,
#         'shift_weight': 0.1,
#         'h_contact_weight': 0.8,
#         'conatct_weight': 0.4
#     }

#     stability_label = compute_stability_score(
#         pos_diff, ori_diff, shift_pos, shift_ori, contacts,
#         pos_diff_max, ori_diff_max, shift_pos_max, shift_ori_max, contacts_max,
#         params=params
#     )
#     return stability_label




class ReplayGrasping:
    def __init__(self, file_path):

        self.file_path = file_path
        self.replay_finished = False
        self.data_logger = None
        self.controller = None
        self.world = None
        self.robot = None
        self.task = None
        self.articulation_controller = None
        self.task_params = None

        self.inputs = None


    def start(self):
        # Set up the world
        self.world: World = World(stage_units_in_meters=1.0)

        # Set up the task
        self.task = PickPlaceCamera()
        self.world.add_task(self.task)

        self.world.reset()

        self.data_logger = self.world.get_data_logger() # a DataLogger object is defined in the World by default
        self.task_params = self.task.get_params()

        self.robot: Franka =  self.world.scene.get_object(self.task_params["robot_name"]["value"])
        self.controller = DataCollectionController(
            name = "data_collection_controller",
            gripper=self.robot.gripper,
            robot_articulation=self.robot
        )
        self.articulation_controller = self.robot.get_articulation_controller() 


        # self.replay_grasping()




    def replay_grasping(self):
        print(f"----------------- Replaying Data: {self.file_path} ----------------- \n")
        asyncio.ensure_future(self._on_replay_scene_event_async(self.file_path))
        return True


    # This is for replying the whole scene
    async def _on_replay_scene_event_async(self, data_file):
            self.data_logger.load(log_path=data_file)

            await self.world.play_async()
            self.world.add_physics_callback("replay_scene", self._on_replay_scene_step)
            return 

    def _on_replay_scene_step(self, step_size):
        print(f"Replaying step {self.world.current_time_step_index} of {self.data_logger.get_num_of_data_frames()}")
        if self.world.current_time_step_index < self.data_logger.get_num_of_data_frames():
            

            cube_name = self.task_params["cube_name"]["value"]
            data_frame = self.data_logger.get_data_frame(data_frame_index=self.world.current_time_step_index)
            self.articulation_controller.apply_action(
                ArticulationAction(joint_positions=data_frame.data["applied_joint_positions"])
            )
            # Sets the world position of the goal cube to the same recoded position
            self.world.scene.get_object(cube_name).set_world_pose(
                position=np.array(data_frame.data["cube_position"]),
                orientation=np.array(data_frame.data["cube_orientation"])
            )
        
            # if data_frame.data["stage"] == 7:
            #     target_position = self.inputs["cube_target_position"]
            #     target_orientation = self.inputs["cube_target_orientation"]
            #     print(f"The target orientation is: {target_orientation}")
            #     angles = quat_to_euler_angles(target_orientation, True)
            #     print(f"And the euler angles are: {angles}")
            #     cube = self.world.scene.add(
            #             DynamicCuboid(
            #                 name="goal",
            #                 position=target_position,
            #                 orientation=target_orientation,
            #                 prim_path="/Cube_final",
            #                 scale=[0.05, 0.05, 0.05],
            #                 size=1.0,
            #                 color=np.array([1, 0, 0]),
            #             )
            #         )
            #     # self.replay_finished = True
            #     self.world.pause()
            #     return 

        elif self.world.current_time_step_index == self.data_logger.get_num_of_data_frames():
            print("----------------- Replay Finished -----------------\n")
            self.replay_finished = True
            
            self.world.remove_physics_callback("replay_scene")
        
        
        return
    



def main():
    # file_path = "/home/chris/Chris/placement_ws/src/random_data/Grasping_115/Placement_27_False.json" # Readlly bad placement
    # file_path = "/home/chris/Chris/placement_ws/src/random_data/Grasping_115/Placement_70_False.json" # Readlly bad placement
    # file_path = "/home/chris/Chris/placement_ws/src/random_data/Grasping_115/Placement_103_False.json" # Readlly bad placement
    # file_path = "/home/chris/Chris/placement_ws/src/random_data/Grasping_115/Placement_22_False.json" # Readlly bad placement
    file_path = "/home/chris/Chris/placement_ws/src/data/benchmark/run_20250224_145419/model_93345942/trajectories/Grasping_1_True.json" # 
    replay_agent = ReplayGrasping(file_path)
    replay_agent.start()
    starting_replay = False
    # data = process_file(file_path)
    # model = single_model_test(3005)
    # pred_score = predict_single_sample(model, data)
    # inputs = data["inputs"]
    # replay_agent.inputs = inputs
    # reg = encode_outputs(data["outputs"])
    # print(f"Your regression label is: {reg}, and the model prediction is: {pred_score}")


    while simulation_app.is_running():
        
        replay_agent.world.step(render=True)
        # print(f"Current time step: {replay_agent.world.current_time_step_index}")
        if replay_agent.world.is_playing():  
            # This function should only be played once
            if not starting_replay:
                starting_replay = True 
                replay_agent.replay_grasping()
                
            if replay_agent.replay_finished:
                # print(f"replay_finished: {replay_agent.world.is_stopped()}")
                starting_replay = False
                replay_agent.replay_finished = False
                

    simulation_app.close()


if __name__ == '__main__':
    main()
