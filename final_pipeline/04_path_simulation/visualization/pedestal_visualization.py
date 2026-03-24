import os
import sys
# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from isaacsim import SimulationApp

DISP_FPS        = 1<<0
DISP_AXIS       = 1<<1
DISP_RESOLUTION = 1<<3
DISP_SKELEKETON   = 1<<9
DISP_MESH       = 1<<10
DISP_PROGRESS   = 1<<11
DISP_DEV_MEM    = 1<<13
DISP_HOST_MEM   = 1<<14

CONFIG = {
    "width": 1920,
    "height":1080,
    "headless": False,
    "renderer": "RayTracedLighting",
    "display_options": DISP_FPS|DISP_RESOLUTION|DISP_MESH|DISP_DEV_MEM|DISP_HOST_MEM,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
}

simulation_app = SimulationApp(CONFIG)


import numpy as np
import json
from omni.isaac.core import World
import omni
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation
from pxr import Sdf, UsdLux


class PedestalVisualization:
    def __init__(self):
        # Table frame: X in [0, 1.50], Y in [-0.30, 0.30], Z up with table plane at Z=0
        # Safe area for pedestal centers with margin m=0.02
        # Pedestal dims (X, Y, Z) = (0.09, 0.11, 0.10)
        self.table_x_min = 0
        self.table_x_max = 0.60
        self.table_y_min = -0.75
        self.table_y_max = 0.75

        self.margin = 0.02
        self.pedestal_dims = np.array([0.27, 0.22, 0.10], dtype=float)
        self.box_dims = np.array([0.143, 0.0915, 0.051], dtype=float)

        # Derived safe center ranges
        half_x = self.pedestal_dims[0] / 2.0
        half_y = self.pedestal_dims[1] / 2.0

        self.safe_x_min = half_x + self.margin                     # 0.045 + 0.02 = 0.065
        self.safe_x_max = self.table_x_max - (half_x + self.margin) # 1.435
        self.safe_y_min = self.table_y_min + (half_y + self.margin) # -0.225
        self.safe_y_max = self.table_y_max - (half_y + self.margin) # +0.225

        # Grid: 8 columns (X), 5 rows (Y)
        self.num_cols = 8
        self.num_rows = 5

        # Scene handles
        self.world = None
        self.pedestals = []
        self.objects = []
        self.robot = None
        self.robot_footprint = None

        # Robot base area defined in table-top coordinates with origin at top-left (0,0):
        # Rectangle corners: (0.6, 0.0) to (0.9, 0.0) to (0.6, 0.15) to (0.9, 0.15)
        # Convert to world coordinates where Y in [-0.30, +0.30] with top edge at +0.30
        self.robot_rect_table = {
            "x_min": 0.6,
            "x_max": 0.9,
            "y_top_min": 0.0,
            "y_top_max": 0.15,
        }
        y_world_max = self.table_y_max - self.robot_rect_table["y_top_min"]  # 0.30 - 0.0 = 0.30
        y_world_min = self.table_y_max - self.robot_rect_table["y_top_max"]  # 0.30 - 0.15 = 0.15
        x_min = self.robot_rect_table["x_min"]
        x_max = self.robot_rect_table["x_max"]
        # Robot sits at the table origin (0,0) with a 0.30×0.30 m footprint
        self.robot_center = np.array([0.0, 0.0], dtype=float)
        self.robot_size   = np.array([0.30, 0.30], dtype=float)

    def _add_light(self):
        """Add a spherical light for visibility (4.2.0 style)."""
        stage = omni.usd.get_context().get_stage()
        sphereLight = UsdLux.SphereLight.Define(stage, Sdf.Path("/World/SphereLight"))
        sphereLight.CreateRadiusAttr(2)
        sphereLight.CreateIntensityAttr(100000)
        XFormPrim(str(sphereLight.GetPath())).set_world_pose([6.5, 0, 12])

    def _hsv_to_rgb(self, h: float, s: float, v: float) -> np.ndarray:
        """Convert HSV in [0,1] to RGB in [0,1]."""
        h6 = h * 6.0
        i = int(np.floor(h6)) % 6
        f = h6 - np.floor(h6)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return np.array([float(r), float(g), float(b)], dtype=float)

    def _generate_distinct_colors(self, n: int) -> list:
        """Generate n distinct RGB colors using golden-ratio hue stepping."""
        if n <= 0:
            return []
        colors = []
        h = 0.0
        golden = 0.6180339887498949
        for k in range(n):
            h = (h + golden) % 1.0
            s = 0.72
            v = 0.95
            colors.append(self._hsv_to_rgb(h, s, v))
        return colors

    def _compute_centers(self):
        """Compute 2×R centers with zero internal gap, centered in the safe span.

        Target rows = 7; if not feasible, fall back to the maximum that fits.
        Columns fixed at 2. Centers overlapping the robot are filtered out.
        """
        # Dimensions and safe spans
        dx = float(self.pedestal_dims[0])
        dy = float(self.pedestal_dims[1])
        lx = float(self.safe_x_max - self.safe_x_min)
        ly = float(self.safe_y_max - self.safe_y_min)

        desired_cols = 2
        desired_rows = 7

        # With zero internal gap: (n-1)*d <= L  =>  n <= floor(L/d) + 1
        max_cols = int(np.floor(lx / dx)) + 1
        max_rows = int(np.floor(ly / dy)) + 1
        n_cols = min(desired_cols, max_cols)
        n_rows = min(desired_rows, max_rows)

        # Center the stack within safe range, leave only edge slack
        x0 = float(self.safe_x_min + 0.5 * (lx - (n_cols - 1) * dx))
        y0 = float(self.safe_y_min + 0.5 * (ly - (n_rows - 1) * dy))
        x_positions = [x0 + k * dx for k in range(n_cols)]
        y_positions = [y0 + k * dy for k in range(n_rows)]

        centers = []
        for i, x in enumerate(x_positions):
            for j, y in enumerate(y_positions):
                if not self._center_overlaps_robot(float(x), float(y)):
                    centers.append((i, j, float(x), float(y)))
        return centers

    def _center_overlaps_robot(self, cx: float, cy: float) -> bool:
        """Return True if pedestal rectangle at (cx,cy) overlaps robot base rectangle."""
        # Pedestal half-extents
        ped_hx = float(self.pedestal_dims[0] / 2.0)
        ped_hy = float(self.pedestal_dims[1] / 2.0)
        # Robot half-extents
        rob_hx = float(self.robot_size[0] / 2.0)
        rob_hy = float(self.robot_size[1] / 2.0)

        # Intervals
        x1_min, x1_max = cx - ped_hx, cx + ped_hx
        y1_min, y1_max = cy - ped_hy, cy + ped_hy
        rx, ry = float(self.robot_center[0]), float(self.robot_center[1])
        x2_min, x2_max = rx - rob_hx, rx + rob_hx
        y2_min, y2_max = ry - rob_hy, ry + rob_hy

        overlap_x = (x1_min <= x2_max) and (x2_min <= x1_max)
        overlap_y = (y1_min <= y2_max) and (y2_min <= y1_max)
        return overlap_x and overlap_y

    def _add_robot(self):
        """Add the Franka robot to the stage for context."""
        robot_prim_path = "/World/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self.robot = Articulation(robot_prim_path)
        self.world.scene.add(self.robot)
        # Place robot at the specified center (Z left at ground level)
        XFormPrim(robot_prim_path).set_world_pose([float(self.robot_center[0]), float(self.robot_center[1]), 0.0])
        # Optional footprint visualization (thin plate)
        self.robot_footprint = VisualCuboid(
            prim_path="/World/robot_footprint",
            name="robot_footprint",
            position=np.array([float(self.robot_center[0]), float(self.robot_center[1]), 0.001]),
            scale=[float(self.robot_size[0]), float(self.robot_size[1]), 0.002],
            color=np.array([0.9, 0.2, 0.2]),
        )
        self.world.scene.add(self.robot_footprint)

    def start(self):
        self.world: World = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
        self.world.scene.add_default_ground_plane()
        self.world.reset()
        self._add_light()
        self._add_robot()

        centers = self._compute_centers()
        colors = self._generate_distinct_colors(len(centers))

        # Pedestal Z center = 0.05 (half height of 0.10)
        pedestal_center_z = float(self.pedestal_dims[2] / 2.0)
        # Object Z center = 0.10 + D_vert/2, with D_vert = box_dims[2] (no rotation)
        object_center_z = float(self.pedestal_dims[2] + self.box_dims[2] / 2.0)

        # Spawn pedestals and objects
        for idx, (i, j, cx, cy) in enumerate(centers):
            p_name = f"pedestal_r{j}_c{i}"
            o_name = f"object_r{j}_c{i}"

            pedestal = VisualCuboid(
                prim_path=f"/World/{p_name}",
                name=p_name,
                position=np.array([cx, cy, pedestal_center_z]),
                scale=self.pedestal_dims.tolist(),
                color=colors[idx],
            )

            obj = VisualCuboid(
                prim_path=f"/World/{o_name}",
                name=o_name,
                position=np.array([cx, cy, object_center_z]),
                scale=self.box_dims.tolist(),
                color=np.array([0.2, 0.4, 0.8]),  # bluish
            )

            self.world.scene.add(pedestal)
            self.world.scene.add(obj)
            self.pedestals.append(pedestal)
            self.objects.append(obj)

        # Save pedestal poses (id, position xyz, orientation quaternion)
        self._save_pedestal_poses(centers, pedestal_center_z)

    def _save_pedestal_poses(self, centers, pedestal_center_z: float) -> None:
        poses = []
        for (i, j, cx, cy) in centers:
            poses.append({
                "id": f"pedestal_r{j}_c{i}",
                "position": [float(cx), float(cy), float(pedestal_center_z)],
                "orientation": [0.0, 0.0, 0.0, 1.0]
            })
        out_path = os.path.join(os.path.dirname(__file__), "pedestal_poses.json")
        print(f"Saving pedestal poses to {out_path}")
        with open(out_path, "w") as f:
            json.dump(poses, f, indent=2)

    def run(self):
        step = 0
        print("\n=== Pedestal Grid Visualization (2×N auto-fit) ===")
        print(f"Safe X range: [{self.safe_x_min:.3f}, {self.safe_x_max:.3f}]")
        print(f"Safe Y range: [{self.safe_y_min:.3f}, {self.safe_y_max:.3f}]\n")
        print(f"Robot base center: ({self.robot_center[0]:.3f}, {self.robot_center[1]:.3f}), size: {self.robot_size.tolist()}")
        print(f"Placed pedestals/objects (after filtering near robot): {len(self.pedestals)}\n")
        while simulation_app.is_running():
            self.world.step(render=True)
            simulation_app.update()
            step += 1


def main():
    env = PedestalVisualization()
    env.start()
    env.run()
    simulation_app.close()


if __name__ == "__main__":
    main()


