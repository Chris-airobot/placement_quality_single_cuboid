import omni
from omni.physx import get_physx_scene_query_interface
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, PhysicsSchemaTools
import carb

class GroundCollisionDetector:
    """
    Detects collisions between objects and a virtual ground using mesh overlap checks.
    """
    
    def __init__(self, stage, non_colliding_part="/World/Franka/panda_link0", ground_height=0.001):
        """
        Initialize the ground collision detector.
        """
        self.stage = stage
        self.ground_height = ground_height
        self.virtual_ground_path = None
        self.colliding_parts = set()  # Track which parts are colliding
        self.non_colliding_part = non_colliding_part
        self.ground_path = "/World/defaultGroundPlane/GroundPlane/CollisionPlane"
        self.obstacle = "/World/CollisionGround"
        self.pedestal = "/World/Pedestal_view"
        self.box_path = "/World/Ycb_object"
        self.excluded_paths = {
            non_colliding_part,
            "/World/Ycb_object",
            "/World/defaultGroundPlane/GroundPlane/CollisionPlane",
            "/World/CollisionGround",
            "/World/Pedestal_view",
            "/World/PreviewBox",
            "/World/Pedestal",
        }
        # Note: Box path is NOT excluded - we want to detect collisions with the box
        
    def create_virtual_ground(self, path="/World/VirtualGround", 
                              size_x=10.0, size_y=10.0, 
                              position=Gf.Vec3f(0, 0, 0),
                              color=Gf.Vec3f(0.2, 0.2, 0.2)):
        """
        Create a thin box that represents the ground visually but has no physics properties.
        """
        self.virtual_ground_path = path
        
        # Create a thin box to represent the ground
        box_geom = UsdGeom.Cube.Define(self.stage, path)
        box_geom.CreateSizeAttr(1.0)
        
        # Set the scale to create a thin ground plane
        scale = Gf.Vec3f(size_x, size_y, self.ground_height)
        box_geom.AddScaleOp().Set(scale)
        
        # Set position and appearance
        box_geom.AddTranslateOp().Set(position)
        box_geom.CreateDisplayColorAttr().Set([color])
        
        return self.stage.GetPrimAtPath(path)
    


    def create_virtual_pedestal(self, path="/World/Pedestal_view", 
                                radius=0.08, height=0.10,
                                position=Gf.Vec3f(0.2, -0.3, 0.05),
                                color=Gf.Vec3f(0.6, 0.6, 0.6)):
        pedestal_geom = UsdGeom.Cylinder.Define(self.stage, path)
        pedestal_geom.CreateRadiusAttr(radius)
        pedestal_geom.CreateHeightAttr(height)
        pedestal_geom.AddTranslateOp().Set(position)
        pedestal_geom.CreateDisplayColorAttr().Set([color])
        return self.stage.GetPrimAtPath(path)
    
    def is_colliding_with_ground(self, robot_part_path):
        """
        Check if a robot part mesh is colliding with the virtual ground using mesh overlap.
        
        Args:
            robot_part_path: USD path to the robot part to check
            
        Returns:
            bool: True if collision detected, False otherwise
        """
        if not self.virtual_ground_path:
            print("Virtual ground not created yet. Call create_virtual_ground first.")
            return False
        
        
        # Get scene query interface
        scene_query = get_physx_scene_query_interface()
        
        # Encode the robot part path for the mesh overlap query
        try:
            # Encode the robot part path for PhysX
            path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.virtual_ground_path))
            
            # This will store our collision result
            collision_detected = [False]
            
            # Callback function for when a hit is detected
            def report_hit(hit):         
                # More flexible comparison approach
                hit_path_str = str(hit.rigid_body)
                # if hit_path_str != self.non_colliding_part and hit_path_str != "/World/Ycb_object" and hit_path_str != self.ground_path\
                #     and hit_path_str != self.obstacle and hit_path_str != self.pedestal:
                if hit_path_str not in self.excluded_paths:
                    collision_detected[0] = True
                    self.colliding_parts.add(hit_path_str)
                    # print(f"Right now, the colliding parts are: {self.colliding_parts}")
                    
                    # Change the color of the virtual ground instead of the robot parts
                    try:
                        ground_geom = UsdGeom.Cube(self.stage.GetPrimAtPath(self.virtual_ground_path))
                        if ground_geom:
                            ground_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red ground
                    except Exception as e:
                        print(f"Error changing ground color: {e}")
                
                return True  # Continue checking other overlaps
            
            # Perform the mesh overlap check
            num_hits = scene_query.overlap_shape(
                path_tuple[0],  # Encoded USD path identifier
                path_tuple[1],  # Encoded instance identifier
                report_hit,           # Callback when hit is found
                False                 # Don't return immediately after first hit
            )
            

            # Reset ground color when no collisions are detected
            if not collision_detected[0]:
                try:
                    ground_geom = UsdGeom.Cube(self.stage.GetPrimAtPath(self.virtual_ground_path))
                    if ground_geom:
                        ground_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.2)])  # Default gray
                except Exception as e:
                    print(f"Error resetting ground color: {e}")
        
            return collision_detected[0]
            
        except Exception as e:
            print(f"Error in mesh overlap detection: {e}")
            return False
        


    def is_colliding_with_pedestal(self, robot_part_path):
        """
        Check if a robot part mesh is colliding with the pedestal using mesh overlap.
        Returns True if collision detected, False otherwise.
        """
        if not self.pedestal:
            print("Pedestal not specified.")
            return False

        scene_query = get_physx_scene_query_interface()

        try:
            path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.pedestal))
            collision_detected = [False]

            def report_hit(hit):
                hit_path_str = str(hit.rigid_body)
                # Exclude the pedestal itself, and (optionally) other exclusions
                if hit_path_str not in self.excluded_paths:
                    # print(f"hitted part: {hit_path_str}")
                    collision_detected[0] = True
                    self.colliding_parts.add(hit_path_str)
                    # Optional: change pedestal color if you want visual feedback
                    try:
                        from pxr import UsdGeom, Gf
                        pedestal_geom = UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.pedestal))
                        if pedestal_geom:
                            pedestal_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red
                    except Exception as e:
                        print(f"Error changing pedestal color: {e}")
                return True

            num_hits = scene_query.overlap_shape(
                path_tuple[0],
                path_tuple[1],
                report_hit,
                False
            )

            # Optional: reset pedestal color if no collision
            if not collision_detected[0]:
                try:
                    from pxr import UsdGeom, Gf
                    pedestal_geom = UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.pedestal))
                    if pedestal_geom:
                        pedestal_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.6)])  # Default gray
                except Exception as e:
                    print(f"Error resetting pedestal color: {e}")

            return collision_detected[0]

        except Exception as e:
            print(f"Error in mesh overlap detection (pedestal): {e}")
            return False

    def is_colliding_with_box(self, robot_part_path):
        """
        Check if a robot part mesh is colliding with the box object using mesh overlap.
        Returns True if collision detected, False otherwise.
        """
        if not self.box_path:
            print("Box path not specified.")
            return False

        scene_query = get_physx_scene_query_interface()

        try:
            path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.box_path))
            collision_detected = [False]

            def report_hit(hit):
                hit_path_str = str(hit.rigid_body)
                # Check all robot parts for collision with the box
                # Exclude the box itself and other non-robot parts
                if hit_path_str not in self.excluded_paths:
                    collision_detected[0] = True
                    self.colliding_parts.add(hit_path_str)
                return True

            num_hits = scene_query.overlap_shape(
                path_tuple[0],
                path_tuple[1],
                report_hit,
                False
            )

            return collision_detected[0]

        except Exception as e:
            print(f"Error in mesh overlap detection (box): {e}")
            return False

    def is_object_colliding_with_pedestal(self):
        """
        Check if the object is colliding with the pedestal using mesh overlap.
        This is specifically for detecting when the grasped object hits the pedestal.
        Returns True if collision detected, False otherwise.
        """
        if not self.box_path or not self.pedestal:
            print("Box path or pedestal not specified.")
            return False

        scene_query = get_physx_scene_query_interface()

        try:
            # Use the pedestal as the query shape to check for overlaps with the object
            path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.pedestal))
            collision_detected = [False]

            def report_hit(hit):
                hit_path_str = str(hit.rigid_body)
                # Check if the object is colliding with the pedestal
                if hit_path_str == self.box_path:
                    collision_detected[0] = True
                    # print(f"hit path: {hit_path_str}")
                    # print(f"⚠️ Object collision with pedestal detected!")
                    # Optional: change pedestal color for visual feedback
                    try:
                        from pxr import UsdGeom, Gf
                        pedestal_geom = UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.pedestal))
                        if pedestal_geom:
                            pedestal_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # Red
                    except Exception as e:
                        print(f"Error changing pedestal color: {e}")
                return True

            num_hits = scene_query.overlap_shape(
                path_tuple[0],
                path_tuple[1],
                report_hit,
                False
            )

            # Reset pedestal color if no collision
            if not collision_detected[0]:
                try:
                    from pxr import UsdGeom, Gf
                    pedestal_geom = UsdGeom.Cylinder(self.stage.GetPrimAtPath(self.pedestal))
                    if pedestal_geom:
                        pedestal_geom.CreateDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.6)])  # Default gray
                except Exception as e:
                    print(f"Error resetting pedestal color: {e}")

            return collision_detected[0]

        except Exception as e:
            print(f"Error in object-pedestal collision detection: {e}")
            return False

    def is_any_part_colliding(self, robot_parts_paths):
        """
        Check if any part of the robot is colliding with the virtual ground.
        """
        return any(self.is_colliding_with_ground(part) for part in robot_parts_paths)

