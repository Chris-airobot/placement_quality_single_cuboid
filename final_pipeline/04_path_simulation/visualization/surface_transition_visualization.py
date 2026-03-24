#!/usr/bin/env python3
"""
Standalone Isaac Sim script for visualizing surface transitions and difficulty metrics.
This script loads labeled data and provides interactive visualization of:
1. Surface transitions with colored faces
2. Difficulty metrics as text overlays
3. Transition arrows between poses
4. Interactive case navigation
"""

import os
import sys
import json
import numpy as np
import datetime
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import random

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from isaacsim import SimulationApp

# Display options for the simulation viewport
DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height": 1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
}

simulation_app = SimulationApp(CONFIG)

import carb
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.objects import VisualCuboid
from pxr import Sdf, UsdLux, Gf, UsdGeom

# --- Surface labeling helper functions ---
def get_surface_up_label(orientation_quat):
    """Determine which surface of the box is facing up"""
    Rmat = R.from_quat(orientation_quat).as_matrix()
    axes = {'x': Rmat[:,0], 'y': Rmat[:,1], 'z': Rmat[:,2]}
    world_up = np.array([0, 0, 1])
    scores = {
        'x_up': np.dot(axes['x'], world_up),
        'x_down': -np.dot(axes['x'], world_up),
        'y_up': np.dot(axes['y'], world_up),
        'y_down': -np.dot(axes['y'], world_up),
        'z_up': np.dot(axes['z'], world_up),
        'z_down': -np.dot(axes['z'], world_up)
    }
    return max(scores, key=scores.get)

def surface_transition_type(init_label, final_label):
    """Determine the type of surface transition"""
    if init_label == final_label:
        return 'same'
    axis = init_label[0]
    if final_label.startswith(axis) and init_label != final_label:
        return 'opposite'
    return 'adjacent'

class SurfaceTransitionVisualizer:
    def __init__(self):
        self.world = None
        self.stage = None
        self.box_dims = np.array([0.143, 0.0915, 0.051])
        self.pedestal_height = 0.10
        
        # Load labeled data
        self.labeled_data_path = '/home/chris/Chris/placement_ws/src/data/box_simulation/v2/run_20250629_180907/labeled_test_data_100_samples.json'
        with open(self.labeled_data_path, 'r') as file:
            self.labeled_data = json.load(file)
        
        # Interactive navigation
        self.current_case_idx = 0
        self.auto_play = False
        self.play_speed = 1.0  # seconds per case
        
        # Visualization objects
        self.initial_object = None
        self.final_object = None
        self.difficulty_indicators = []
        
        # Surface colors
        self.surface_colors = {
            'z_up': np.array([1.0, 0.0, 0.0]),    # Red
            'z_down': np.array([0.0, 1.0, 0.0]),  # Green  
            'x_up': np.array([0.0, 0.0, 1.0]),    # Blue
            'x_down': np.array([1.0, 1.0, 0.0]),  # Yellow
            'y_up': np.array([1.0, 0.0, 1.0]),    # Magenta
            'y_down': np.array([0.0, 1.0, 1.0])   # Cyan
        }
        
        # Transition colors
        self.transition_colors = {
            'same': np.array([0.0, 1.0, 0.0]),      # Green
            'adjacent': np.array([1.0, 1.0, 0.0]),  # Yellow  
            'opposite': np.array([1.0, 0.0, 0.0])   # Red
        }

    def setup_scene(self):
        """Set up the Isaac Sim scene"""
        create_new_stage()
        self.stage = omni.usd.get_context().get_stage()
        self._add_lighting()
        
        self.world = World()
        self.world.scene.add_default_ground_plane()
        
        # Create pedestal
        self._create_pedestal()
        
        # Create visualization objects
        self._create_visualization_objects()
        
        self.world.reset()
        print("Scene setup complete")

    def _add_lighting(self):
        """Add lighting to the scene"""
        sphereLight = UsdLux.SphereLight.Define(self.stage, Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _create_pedestal(self):
        """Create a pedestal for the objects"""
        from omni.isaac.core.objects import VisualCylinder
        
        pedestal = VisualCylinder(
            prim_path="/World/Pedestal",
            name="Pedestal",
            position=np.array([0.2, -0.3, self.pedestal_height/2]),
            radius=0.08,
            height=self.pedestal_height,
            color=np.array([0.6, 0.6, 0.6])
        )
        self.world.scene.add(pedestal)

    def _create_visualization_objects(self):
        """Create the objects for visualization"""
        # Initial object (left side)
        self.initial_object = VisualCuboid(
            prim_path="/World/InitialObject",
            name="InitialObject",
            position=np.array([0.0, -0.3, self.pedestal_height + self.box_dims[2]/2]),
            scale=self.box_dims.tolist(),
            color=np.array([0.8, 0.8, 0.8])
        )
        
        # Final object (right side)
        self.final_object = VisualCuboid(
            prim_path="/World/FinalObject", 
            name="FinalObject",
            position=np.array([0.4, -0.3, self.pedestal_height + self.box_dims[2]/2]),
            scale=self.box_dims.tolist(),
            color=np.array([0.8, 0.8, 0.8])
        )
        
        # Create difficulty indicators (colored spheres)
        self._create_difficulty_indicators()
        
        self.world.scene.add(self.initial_object)
        self.world.scene.add(self.final_object)

    def _create_difficulty_indicators(self):
        """Create visual indicators for difficulty metrics"""
        from omni.isaac.core.objects import VisualSphere
        
        # Create spheres for each difficulty metric
        indicator_positions = [
            [0.2, -0.5, 0.25],  # Angular difficulty
            [0.2, -0.5, 0.20],  # Joint difficulty  
            [0.2, -0.5, 0.15],  # Manipulability difficulty
        ]
        
        self.difficulty_indicators = []
        for i, pos in enumerate(indicator_positions):
            indicator = VisualSphere(
                prim_path=f"/World/DifficultyIndicator_{i}",
                name=f"DifficultyIndicator_{i}",
                position=np.array(pos),
                radius=0.02,
                color=np.array([0.5, 0.5, 0.5])  # Default gray
            )
            self.world.scene.add(indicator)
            self.difficulty_indicators.append(indicator)

    def _color_object_by_surface(self, object_prim, surface_label):
        """Color the object based on which surface is up"""
        if surface_label in self.surface_colors:
            color = self.surface_colors[surface_label]
            # Set the display color
            geom = UsdGeom.Gprim(object_prim.prim)
            geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def _update_difficulty_indicators(self, case):
        """Update the difficulty indicators with colors"""
        # Difficulty color mapping
        difficulty_colors = {
            'Easy': np.array([0.0, 1.0, 0.0]),    # Green
            'Medium': np.array([1.0, 1.0, 0.0]),  # Yellow
            'Hard': np.array([1.0, 0.0, 0.0])     # Red
        }
        
        # Update each indicator
        difficulties = [
            case['difficulty_label_angular'],
            case['difficulty_label_joint'], 
            case['difficulty_label_manipulability']
        ]
        
        for i, (indicator, difficulty) in enumerate(zip(self.difficulty_indicators, difficulties)):
            if difficulty in difficulty_colors:
                color = difficulty_colors[difficulty]
                geom = UsdGeom.Gprim(indicator.prim)
                geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def _draw_transition_arrow(self, start_pos, end_pos, transition_type):
        """Draw an arrow showing the transition between poses"""
        from omni.isaac.debug_draw import _debug_draw
        from carb import Float3, ColorRgba
        
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_lines()
        
        if transition_type in self.transition_colors:
            color = self.transition_colors[transition_type]
            carb_color = ColorRgba(color[0], color[1], color[2], 1.0)
            
            start_point = Float3(*start_pos)
            end_point = Float3(*end_pos)
            
            draw.draw_lines([start_point], [end_point], [carb_color], [3.0])

    def display_case(self, case_idx):
        """Display a specific case"""
        if case_idx < 0 or case_idx >= len(self.labeled_data):
            return
        
        self.current_case_idx = case_idx
        case = self.labeled_data[case_idx]
        
        # Get poses
        initial_pose = case['initial_object_pose']
        final_pose = case['final_object_pose']
        
        # Update object positions and orientations
        initial_pos = np.array([0.0, -0.3, self.pedestal_height + self.box_dims[2]/2])
        final_pos = np.array([0.4, -0.3, self.pedestal_height + self.box_dims[2]/2])
        
        self.initial_object.set_world_pose(
            position=initial_pos,
            orientation=initial_pose[3:]
        )
        
        self.final_object.set_world_pose(
            position=final_pos,
            orientation=final_pose[3:]
        )
        
        # Color objects based on surface labels
        self._color_object_by_surface(self.initial_object, case['surface_up_initial'])
        self._color_object_by_surface(self.final_object, case['surface_up_final'])
        
        # Update difficulty indicators
        self._update_difficulty_indicators(case)
        
        # Draw transition arrow
        self._draw_transition_arrow(
            initial_pos,
            final_pos,
            case['surface_transition_type']
        )
        
        # Print case information to console
        print(f"\n=== Case {case_idx + 1}/{len(self.labeled_data)} ===")
        print(f"Surface Transition: {case['surface_up_initial']} -> {case['surface_up_final']} ({case['surface_transition_type']})")
        print(f"Angular Difficulty: {case['difficulty_label_angular']}")
        print(f"Joint Difficulty: {case['difficulty_label_joint']}")
        print(f"Manipulability Difficulty: {case['difficulty_label_manipulability']}")
        print(f"Success: {case['success_label']}")
        print("Visual Indicators:")
        print("  - Colored faces show which surface is 'up'")
        print("  - Colored spheres: Green=Easy, Yellow=Medium, Red=Hard")
        print("  - Arrow color: Green=Same, Yellow=Adjacent, Red=Opposite")

    def next_case(self):
        """Go to next case"""
        if self.current_case_idx < len(self.labeled_data) - 1:
            self.display_case(self.current_case_idx + 1)

    def prev_case(self):
        """Go to previous case"""
        if self.current_case_idx > 0:
            self.display_case(self.current_case_idx - 1)

    def toggle_auto_play(self):
        """Toggle auto-play mode"""
        self.auto_play = not self.auto_play
        print(f"Auto-play: {'ON' if self.auto_play else 'OFF'}")

    def run(self):
        """Main run loop"""
        self.setup_scene()
        
        # Display first case
        self.display_case(0)
        
        print("\n=== Surface Transition Visualizer ===")
        print("Controls:")
        print("  N - Next case")
        print("  P - Previous case") 
        print("  A - Toggle auto-play")
        print("  Q - Quit")
        print("  Number keys (0-9) - Jump to case")
        print("\nVisual Legend:")
        print("  Face Colors: Red=Z-up, Green=Z-down, Blue=X-up, Yellow=X-down, Magenta=Y-up, Cyan=Y-down")
        print("  Difficulty Spheres: Green=Easy, Yellow=Medium, Red=Hard")
        print("  Transition Arrows: Green=Same, Yellow=Adjacent, Red=Opposite")
        print("=====================================\n")
        
        last_auto_play_time = 0
        
        while simulation_app.is_running():
            # Handle keyboard input
            if carb.input.is_keyboard_event():
                event = carb.input.get_keyboard_event()
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if event.input == carb.input.KeyboardInput.N:
                        self.next_case()
                    elif event.input == carb.input.KeyboardInput.P:
                        self.prev_case()
                    elif event.input == carb.input.KeyboardInput.A:
                        self.toggle_auto_play()
                    elif event.input == carb.input.KeyboardInput.Q:
                        break
                    elif event.input in [carb.input.KeyboardInput.NUM_0, carb.input.KeyboardInput.NUM_1,
                                       carb.input.KeyboardInput.NUM_2, carb.input.KeyboardInput.NUM_3,
                                       carb.input.KeyboardInput.NUM_4, carb.input.KeyboardInput.NUM_5,
                                       carb.input.KeyboardInput.NUM_6, carb.input.KeyboardInput.NUM_7,
                                       carb.input.KeyboardInput.NUM_8, carb.input.KeyboardInput.NUM_9]:
                        # Jump to case based on number key
                        case_num = int(event.input) - int(carb.input.KeyboardInput.NUM_0)
                        if case_num == 0:
                            case_num = 10
                        target_case = (case_num - 1) * 10  # Jump to cases 0, 10, 20, etc.
                        if target_case < len(self.labeled_data):
                            self.display_case(target_case)
            
            # Auto-play logic
            if self.auto_play:
                current_time = carb.get_time()
                if current_time - last_auto_play_time > self.play_speed:
                    self.next_case()
                    last_auto_play_time = current_time
            
            # Step simulation
            self.world.step(render=True)
            simulation_app.update()

if __name__ == "__main__":
    visualizer = SurfaceTransitionVisualizer()
    visualizer.run()
    simulation_app.close()
