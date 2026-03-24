from difficulty_labeling import DifficultyLabeler

test_data = [
    # Example 1: Identity (no change)
    {
        'grasp_pose': [0, 0, 0, 0, 0, 0, 1],
        'initial_object_pose': [0, 0, 0, 1, 0, 0, 0],
        'final_object_pose':   [0, 0, 0, 1, 0, 0, 0],
        'success_label': 1.0,
        'collision_label': 1.0
    },
    # Example 2: 90-degree yaw (around Z-axis)
    {
        'grasp_pose': [0, 0, 0, 0, 0, 0, 1],
        'initial_object_pose': [0, 0, 0, 1, 0, 0, 0],
        'final_object_pose':   [0, 0, 0, 0.7071, 0, 0, 0.7071],  # 90° Z-rotation
        'success_label': 1.0,
        'collision_label': 1.0
    },
    # Example 3: 180-degree flip (around Y-axis) - CORRECTED
    {
        'grasp_pose': [0, 0, 0, 1, 0, 0, 0],
        'initial_object_pose': [0, 0, 0, 1, 0, 0, 0],
        'final_object_pose':   [0, 0, 0, 0, 0, 1, 0],  # 180° Y-rotation
        'success_label': 1.0,
        'collision_label': 1.0
    },
    # Example 4: 180-degree flip (around X-axis)
    {
        'grasp_pose': [0, 0, 0, 1, 0, 0, 0],
        'initial_object_pose': [0, 0, 0, 1, 0, 0, 0],
        'final_object_pose':   [0, 0, 0, 0, 1, 0, 0],  # 180° X-rotation
        'success_label': 1.0,
        'collision_label': 1.0
    }
]

difficulty_labeler = DifficultyLabeler()
difficulty_labeler.setup_scene()

print("=== SANITY CHECK RESULTS ===\n")

for i, current_data in enumerate(test_data):
    print(f"Test Case {i+1}:")
    
    initial_quat = current_data['initial_object_pose'][3:]
    final_quat = current_data['final_object_pose'][3:]
    
    angular_distance = difficulty_labeler.calculate_angular_distance(initial_quat, final_quat)
    joint_distance, final_joints = difficulty_labeler.calculate_joint_distance(
        current_data['grasp_pose'], 
        current_data['initial_object_pose'], 
        current_data['final_object_pose']
    )
    if final_joints is None:
        manipulability = None
    else:
        manipulability = difficulty_labeler.compute_jacobian(final_joints)
    
    print(f"  Angular distance: {angular_distance:.1f}°")
    print(f"  Joint distance: {joint_distance:.4f}")
    print(f"  Manipulability: {manipulability:.6f}")
    
    # Expected values
    if i == 0:  # Identity
        print(f"  Expected: 0°, ~0, ~0.25")
    elif i == 1:  # 90° Z-rotation
        print(f"  Expected: 90°, ~2-6, ~0.1")
    elif i == 2:  # 180° Y-rotation
        print(f"  Expected: 180°, ~3-8, ~0.15")
    elif i == 3:  # 180° X-rotation
        print(f"  Expected: 180°, ~3-8, ~0.15")
    
    print()