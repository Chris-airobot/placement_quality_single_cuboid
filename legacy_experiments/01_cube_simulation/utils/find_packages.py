"""
All you need to do is simply import the package you want to find for isaac sim, and put it into the "modules"
"""


# import carb
from isaacsim import SimulationApp
import sys

BACKGROUND_STAGE_PATH = "/background"
BACKGROUND_USD_PATH = "/home/chris/Chris/placement_ws/src/grasp_placement/panda.usd"

CONFIG = {"renderer": "RayTracedLighting", "headless": True}

# Example ROS2 bridge sample demonstrating the manual loading of stages and manual publishing of images
simulation_app = SimulationApp(CONFIG)

import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata as sd
import sys
import carb
import warp

modules = [
    "carb",
    "omni.isaac.core.utils",
    "omni.isaac.core_nodes",
    "warp"
]

for module_name in modules:
    if module_name in sys.modules:
        print(f"\nModule '{module_name}' is loaded.")
        mod = sys.modules[module_name]
        # Attempt to print the __file__ attribute if it exists
        try:
            print(f"  __file__ = {mod.__path__}")
        except AttributeError:
            print("  No __file__ attribute (likely a compiled/binary extension).")
    else:
        print(f"\nModule '{module_name}' is NOT loaded.")



