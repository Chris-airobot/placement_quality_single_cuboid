from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World

from omni.isaac.core.utils import extensions
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.franka import Franka
from controllers.pick_place_task_with_camera import PickPlaceCamera
from controllers.data_collection_controller import DataCollectionController
from omni.isaac.core.utils.types import ArticulationAction
from helper import *
from omni.isaac.sensor import ContactSensor
from utilies.camera_utility import *
from omni.isaac.dynamic_control import _dynamic_control
from carb import Float3
import random
from functools import partial
from typing import List
# Enable ROS2 bridge extension
extensions.enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()


my_world: World = World(stage_units_in_meters=1.0)
my_task = PickPlaceCamera(set_camera=False)
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_franka: Franka = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = DataCollectionController(
    name="pick_place_controller", gripper=my_franka.gripper, robot_articulation=my_franka
)
articulation_controller = my_franka.get_articulation_controller()
# Acquire the dynamic control interface
dc = _dynamic_control.acquire_dynamic_control_interface()

pedestrian1_handle = dc.get_rigid_body("/World/Cube")


panda_prim_names = [
            "Franka/panda_link0",
            "Franka/panda_link1",
            "Franka/panda_link2",
            "Franka/panda_link3",
            "Franka/panda_link4",
            "Franka/panda_link5",
            "Franka/panda_link6",
            "Franka/panda_link7",
            "Franka/panda_link8",
            "Franka/panda_hand",
            "Cube"
        ]

contact_sensors = []
for i, link_name in enumerate(panda_prim_names):
    sensor: ContactSensor = my_world.scene.add(
        ContactSensor(
            prim_path=f"/World/{link_name}/contact_sensor",
            name=f"contact_sensor_{i}",
            min_threshold=0.0,
            max_threshold=1e7,
            radius=0.1,
        )
    )
    # Use raw contact data if desired
    sensor.add_raw_contact_data_to_frame()
    contact_sensors.append(sensor)

my_world.reset()

def sensor_cb(dt, sensors: List[ContactSensor]):
    """Physics-step callback: checks all sensors, sets self.contact accordingly."""
    any_contact = False  # track if at least one sensor had contact
    cube_contacted = False
    cube_grasped = None

    for sensor in sensors:
        frame_data = sensor.get_current_frame()
        if frame_data["in_contact"]:
            # We have contact! Extract the bodies, force, etc.
            for c in frame_data["contacts"]:
                body0 = c["body0"]
                body1 = c["body1"]
                if "panda" in body0 + body1 and "Cube" in body0 + body1:
                    # print("Cube in the ground")
                    cube_grasped = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"

                if "GroundPlane" in body0 + body1 and "Cube" in body0 + body1:
                    # print("Cube in the ground")
                    cube_contacted = True
                elif ("GroundPlane" in body0) or ("GroundPlane" in body1):
                    print("Robot hits the ground, and it will be recorded")
                    any_contact = True
                    contact = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"
                    # self.contact = f"{body0} | {body1} | Force: {frame_data['force']:.3f} | #Contacts: {frame_data['number_of_contacts']}"
    # print(f"cube is in the ground: {self.cube_contacted}, time is {self.world.current_time_step_index}, stage is {self.controller.get_current_event()}")

    # If, after checking all sensors, none had contact, reset self.contact to None
    if not any_contact:
        contact = None

# Contact report for links
my_world.add_physics_callback("contact_sensor_callback", partial(sensor_cb, sensors=contact_sensors))















reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0.005, 0]),
        )
        possible_forces = [Float3(5, 0, 0), Float3(0, 5, 0), Float3(0, 0, 5)]
        force_vec = random.choice(possible_forces)

        if my_controller.get_current_event() > 4 and my_controller.get_current_event() < 7:
            position_vec = Float3(np.random.uniform(0, 5, 3))
            dc.apply_body_force(pedestrian1_handle, force_vec, position_vec, False)

        articulation_controller.apply_action(actions)
simulation_app.close()