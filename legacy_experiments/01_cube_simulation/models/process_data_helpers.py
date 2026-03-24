import os
import json
import re
import sys
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.axangles import axangle2mat
import numpy as np
from scipy.spatial.transform import Rotation as R
import tf_transformations as tft 
import matplotlib.pyplot as plt
import math
import random
# Add the parent folder to sys.path so `learning_models` is found
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from models.dataset import MyStabilityDataset

def data_split(data_path="/home/chris/Chris/placement_ws/src/data/processed_data/data.json"):
    # Define paths
    output_dir = "/home/chris/Chris/placement_ws/src/data/processed_data/data_splits"
    os.makedirs(output_dir, exist_ok=True)

    # Load the full dataset
    with open(data_path, "r") as file:
        all_entries = json.load(file)

    # Shuffle the dataset to randomize the entries
    random.shuffle(all_entries)

    # Compute split sizes
    N = len(all_entries)
    train_size = int(0.8 * N)
    valid_size = int(0.1 * N)
    # Ensure all entries are used for test set
    test_size = N - train_size - valid_size

    # Split the data
    train_entries = all_entries[:train_size]
    valid_entries = all_entries[train_size:train_size + valid_size]
    test_entries = all_entries[train_size + valid_size:]

    # Save each split to a separate file
    with open(os.path.join(output_dir, "train_data.json"), "w") as f:
        json.dump(train_entries, f, indent=4)

    with open(os.path.join(output_dir, "valid_data.json"), "w") as f:
        json.dump(valid_entries, f, indent=4)

    with open(os.path.join(output_dir, "test_data.json"), "w") as f:
        json.dump(test_entries, f, indent=4)

    print(f"Data split completed:\n - Train/Validation/Test: {len(train_entries)} / {len(valid_entries)} / {len(test_entries)}")





def merge_json_files(file1, file2, output_file):
    """
    Merge two JSON files into a single file.
    """
    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    combined_data = data1 + data2  # Directly concatenates the two lists

    with open(output_file, 'w') as fout:
        json.dump(combined_data, fout, indent=4)


def count_files_in_subfolders(directory):
    """
    Count the number of files within all subfolders of a directory.

    Args:
        directory (str): Path to the main directory.

    Returns:
        dict: A dictionary where keys are subfolder paths and values are the file counts.
        int: Total number of files across all subfolders.
    """
    file_count_per_subfolder = {}
    total_file_count = 0

    for root, dirs, files in os.walk(directory):
        # Only consider subfolders (not the main folder)
        if root != directory:
            file_count = len(files)
            file_count_per_subfolder[root] = file_count
            total_file_count += file_count


    sorted_subfolders = sorted(
        ((subfolder, count) for subfolder, count in file_count_per_subfolder.items() if count > 1 
        and os.path.basename(subfolder).startswith("Grasping_")),
        key=lambda item: int(os.path.basename(item[0]).split('_')[-1])
    )

    for subfolder, count in sorted_subfolders:
        print(f"{subfolder}: {count} files")

    print(f"Total files across all subfolders: {total_file_count}")


def rough_analysis(file_path):
    # Load your JSON file
    with open(file_path, "r") as file:
        trajectories = json.load(file)
    
    # Count trajectories with at least one null (None) value in the outputs
    null_trajectory_count = sum(
        1 for traj in trajectories 
        if any(value is None for value in traj.get("outputs", {}).values())
    )

    unsuccessful_count = sum(
        1 for traj in trajectories 
        if traj.get("outputs", {}).get("grasp_unsuccessful") is True
    )
    print(f"Total number of trajectories: {len(trajectories)}")
    print("Number of trajectories with null outputs:", null_trajectory_count)
    print("Number of trajectories with unsuccessful grasp:", unsuccessful_count)


def data_analysis(file_path, split=False):
    with open(file_path, "r") as file:
        all_entries = json.load(file)

        
    pos_diffs   = [x["outputs"]["position_difference"] for x in all_entries if x["outputs"]["position_difference"] is not None]
    ori_diffs   = [x["outputs"]["orientation_difference"] for x in all_entries if x["outputs"]["orientation_difference"] is not None]
    shift_poss  = [x["outputs"]["shift_position"]       for x in all_entries if x["outputs"]["shift_position"] is not None]
    shift_oris  = [x["outputs"]["shift_orientation"]    for x in all_entries if x["outputs"]["shift_orientation"] is not None]
    contacts_ls = [x["outputs"]["contacts"]            for x in all_entries if x["outputs"]["contacts"] is not None]

    # Convert to np arrays to avoid the ValueError
    pos_diffs   = np.array(pos_diffs,   dtype=float)
    ori_diffs   = np.array(ori_diffs,   dtype=float)
    shift_poss  = np.array(shift_poss,  dtype=float)
    shift_oris  = np.array(shift_oris,  dtype=float)
    contacts_ls = np.array(contacts_ls, dtype=float)

    pos_diff_95p      = np.percentile(pos_diffs, 90)
    ori_diff_95p      = np.percentile(ori_diffs, 90)
    shift_pos_95p     = np.percentile(shift_poss, 90)
    shift_ori_95p     = np.percentile(shift_oris, 90)
    contacts_95p      = np.percentile(contacts_ls, 90)
    
    # Filter out outliers (remove any values above the 95th percentile)
    pos_diffs_f   = pos_diffs[pos_diffs <= np.percentile(pos_diffs, 95)]
    ori_diffs_f   = ori_diffs[ori_diffs <= np.percentile(ori_diffs, 95)]
    shift_poss_f  = shift_poss[shift_poss <= np.percentile(shift_poss, 95)]
    shift_oris_f  = shift_oris[shift_oris <= np.percentile(shift_oris, 95)]
    contacts_ls_f = contacts_ls[contacts_ls <= np.percentile(contacts_ls, 95)]



    print(f"values are:pose_diffs: {pos_diff_95p}, ori_diffs: {ori_diff_95p}, shift_poss: {shift_pos_95p}, shift_oris: {shift_ori_95p}, contacts: {contacts_95p}")
    output_file = "/home/chris/Chris/placement_ws/src/data/processed_data/parameters.json"
    combined_data = {
        "position_difference": pos_diff_95p,
        "orientation_difference": ori_diff_95p,
        "shift_position": shift_pos_95p,
        "shift_orientation": shift_ori_95p,
        "contacts": contacts_95p,
        "max_score": None,
        "min_score": None
    }

    with open(output_file, 'w') as fout:
        json.dump(combined_data, fout, indent=4)

    
    whole_dataset = MyStabilityDataset(all_entries)

    stability_scores = []
    for _, labels in whole_dataset:
        # labels is a tuple (feasibility_label, stability_label)
        stability_scores.append(labels[1].item())  # Convert tensor to Python float
    
    stability_scores = np.array(stability_scores)
    q1 = np.percentile(stability_scores, 25)  # First quartile (Q1)
    q3 = np.percentile(stability_scores, 75)  # Third quartile (Q3)
    iqr = q3 - q1  # Interquartile range
    lower_bound = q1 - 1.5 * iqr  # Lower bound
    upper_bound = q3 + 1.5 * iqr  # Upper bound
    
    # print(f"before filtering: {len(stability_scores)}")
    updated = [x for x in stability_scores if lower_bound <= x <= upper_bound]
    # print(f"after filtering: {len(updated)}")

    combined_data["max_score"] = max(updated)
    combined_data["min_score"] = min(updated)

    with open(output_file, 'w') as fout:
        json.dump(combined_data, fout, indent=4)

    if split:
        data_split()

    # Create a figure with 2 rows x 3 columns (6 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()  # Flatten the axes array for easier indexing

    # Plot 1: Position Difference (Filtered)
    x_idx = np.arange(len(pos_diffs_f))
    # axes[0].scatter(x_idx, pos_diffs_f, color='skyblue')
    axes[0].hist(pos_diffs_f, bins=20, color='skyblue', edgecolor='k')
    axes[0].axhline(pos_diff_95p, color='red', linestyle='--', label='95th Percentile')
    axes[0].set_title("Position Difference (Filtered)")
    axes[0].set_xlabel("Pos_diff (meters)")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Plot 2: Orientation Difference (Filtered)
    x_idx = np.arange(len(ori_diffs_f))
    axes[1].hist(ori_diffs_f, bins=20, color='skyblue', edgecolor='k')
    axes[1].axhline(ori_diff_95p, color='red', linestyle='--', label='95th Percentile')
    axes[1].set_title("Orientation Difference (Filtered)")
    axes[1].set_xlabel("Ori_diff (radians)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    # Plot 3: Shift Position (Filtered)
    x_idx = np.arange(len(shift_poss_f))
    axes[2].hist(shift_poss_f, bins=20, color='skyblue', edgecolor='k')
    axes[2].axhline(shift_pos_95p, color='red', linestyle='--', label='95th Percentile')
    axes[2].set_title("Shift Position (Filtered)")
    axes[2].set_xlabel("Shift_pos (meters)")
    axes[2].set_ylabel("Frequency")
    axes[2].legend()

    # Plot 4: Shift Orientation (Filtered)
    x_idx = np.arange(len(shift_oris_f))
    axes[3].hist(shift_oris_f, bins=20, color='skyblue', edgecolor='k')
    axes[3].axhline(shift_ori_95p, color='red', linestyle='--', label='95th Percentile')
    axes[3].set_title("Shift Orientation (Filtered)")
    axes[3].set_xlabel("Shift_ori (radians)")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()

    # Plot 5: Contacts (Filtered)
    x_idx = np.arange(len(contacts_ls_f))
    axes[4].hist(contacts_ls_f, bins=20, color='skyblue', edgecolor='k')

    axes[4].axhline(contacts_95p, color='red', linestyle='--', label='95th Percentile')
    axes[4].set_title("Contacts (Filtered)")
    axes[4].set_xlabel("Contacts")
    axes[4].set_ylabel("Frequency")
    axes[4].legend()

    # Hide the 6th subplot (or use it for a summary)
    axes[5].axis("off")

    plt.tight_layout()
    plt.show()

    
def convert_wxyz_to_xyzw(q_wxyz):
    """Convert a quaternion from [w, x, y, z] format to [x, y, z, w] format."""
    return [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]


def get_transform_to_world(tf_list, target_frame):
    """
    Recursively computes the transform from the 'world' frame to target_frame.
    
    Each TF entry in tf_list is assumed to have:
      - "parent_frame": The parent frame name.
      - "child_frame": The child frame name.
      - "translation": A dict with keys "x", "y", "z".
      - "rotation": A dict with keys "x", "y", "z", "w" (in wxyz order in the data).
    
    Returns a 4x4 homogeneous transformation matrix representing the pose of 
    target_frame in the world coordinate system, or None if no chain can be built.
    """
    if target_frame == "world":
        return np.eye(4)
    
    # Look for an entry whose child_frame is the target_frame.
    for tf_entry in tf_list:
        if tf_entry.get("child_frame") == target_frame:
            # Get the transform from its parent to this target_frame.
            translation = tf_entry["translation"]
            rotation = tf_entry["rotation"]
            pos = np.array([translation["x"], translation["y"], translation["z"]])
            # Convert quaternion from the tf data order (wxyz) to (x,y,z,w) order.
            quat_wxyz = np.array([rotation["w"], rotation["x"], rotation["y"], rotation["z"]])
            T_target_given_parent = pose_to_homogeneous(pos, quat_wxyz)
            
            parent_frame = tf_entry["parent_frame"]
            # Recursively get the transform from world to the parent_frame.
            T_parent = get_transform_to_world(tf_list, parent_frame)
            if T_parent is None:
                # Could not resolve parent transform.
                return None
            # The complete transform is the chain: T_world->target = T_world->parent * T_parent->target.
            return np.dot(T_parent, T_target_given_parent)
    # If no entry is found for the target_frame, return None.
    return None

def get_relative_transform(tf_list, source_frame, target_frame):
    """
    Computes the relative transform from source_frame to target_frame.
    
    The transformation is computed as:
       T_target_in_source = (T_source_in_world)^{-1} * T_target_in_world
       
    Returns the 4x4 homogeneous transformation matrix representing the pose 
    of target_frame in the coordinate system of source_frame.
    """
    T_source = get_transform_to_world(tf_list, source_frame)
    T_target = get_transform_to_world(tf_list, target_frame)
    
    if T_source is None:
        print(f"Transform chain to source frame '{source_frame}' could not be resolved.")
        return None
    if T_target is None:
        print(f"Transform chain to target frame '{target_frame}' could not be resolved.")
        return None
    
    # Compute the relative transform.
    T_relative = np.dot(np.linalg.inv(T_source), T_target)
    return T_relative


def transform_pose_to_frame(world_position, world_orientation, T_frame_inv):
    """
    Transform a world-frame pose (position & orientation in wxyz) into a new frame.
    Returns (relative_position, relative_orientation in wxyz).
    """
    T_world = pose_to_homogeneous(world_position, world_orientation)
    T_rel = np.dot(T_frame_inv, T_world)
    rel_pos = T_rel[0:3, 3].tolist()
    rel_orient = tft.quaternion_from_matrix(T_rel)
    return rel_pos, [rel_orient[3], rel_orient[0], rel_orient[1], rel_orient[2]]





def homogeneous_to_pose(T):
    """
    Convert a 4x4 transform to (position, orientation).
    Orientation is [w, x, y, z].
    """
    # position
    position = T[:3, 3].tolist()

    # tf.transformations.quaternion_from_matrix returns [x,y,z,w].
    q_xyzw = tft.quaternion_from_matrix(T)  # x,y,z,w

    # Reorder to [w, x, y, z]
    orientation = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
    return position, orientation


def pose_to_homogeneous(position_xyz, orientation_xyzw):
    """
    Convert a pose to a 4x4 transform.
      position_xyz: [x,y,z]
      orientation_xyzw: quaternion in [x,y,z,w]
    """
    T = np.eye(4)
    T[:3, 3] = position_xyz
    R = tft.quaternion_matrix(orientation_xyzw)  # returns 4×4
    T[:3, :3] = R[:3, :3]
    return T




def pose_difference(final_pos, final_quat, target_pos, target_quat, exclude_z = False):
    # 1. Position difference
    d_pos = np.linalg.norm(np.array(final_pos) - np.array(target_pos)) if not exclude_z else np.linalg.norm(np.array(final_pos)[:2] - np.array(target_pos)[:2])

    # 2. Orientation difference
    # relative_quat = target_quat * inverse(final_quat)
    # Then angle = 2 * acos(|relative_quat.w|)
    relative_quat = qmult(target_quat, qinverse(final_quat))

    norm = np.sqrt(sum(q*q for q in relative_quat))
    normalize = [q / norm for q in relative_quat]


    w = abs(normalize[0])
    w = np.clip(w, -1.0, 1.0) 
    angle = 2.0 * np.arccos(w)  # [0] is 'w' if your quat is [w, x, y, z]

    return float(d_pos), float(angle)


def process_file(file_path):
    """
    Process a single file to extract relevant information for training.

    """
    with open(file_path, "r") as file:
        data_all = json.load(file)

    isaac_sim_data = data_all["Isaac Sim Data"]
    if not isaac_sim_data:
        raise ValueError(f"No data entries found under 'Isaac Sim Data' in {file_path}")
    

    # Match the first boolean after "placement_XX_"
    match = re.search(r"Placement_\d+_(False|True)", file_path)
    grasp_unsuccessful = (match.group(1) == "True") if match else None

    # Initialize variables
    contact_count = 0

    ###############################
    ####### INPUT EXTRACTION
    ###############################
        # --- Assemble inputs (return both world-frame and relative poses as desired) ---
    inputs = {
        "grasp_position": None,
        "grasp_orientation": None,
        "cube_initial_position": isaac_sim_data[0]["data"]["cube_position"],
        "cube_initial_orientation": isaac_sim_data[0]["data"]["cube_orientation"],
        "cube_target_position": isaac_sim_data[0]["data"]["target_position"],
        "cube_target_orientation": list(euler2quat(*np.random.uniform(low=-np.pi, high=np.pi, size=3))),
        # "cube_target_surface": int(np.random.randint(1, 7)),
        "cube_initial_rel_position": None,          # Relative to the gripper frame when the robot just grasped the cube.
        "cube_initial_rel_orientation": None,       # Relative to the gripper frame when the robot just grasped the cube.
        "cube_target_rel_position": None,           # Relative to the gripper frame when the robot just grasped the cube.
        "cube_target_rel_orientation": None,        # Relative to the gripper frame when the robot just grasped the cube.
    }

    
    T_grasp = None
    # Find the entry with the maximum current_time
    if not grasp_unsuccessful:
        stage4 = [d for d in isaac_sim_data if d["data"]["stage"] == 4]
        first_stage4 = min(stage4, key=lambda x: x["current_time"])
        data_after_stage4 = [entry for entry in isaac_sim_data if entry["current_time"] >= first_stage4["current_time"]]
        inputs["grasp_position"] = first_stage4["data"]["ee_position"]
        inputs["grasp_orientation"] = first_stage4["data"]["ee_orientation"]
        inputs["cube_initial_position"] = first_stage4["data"]["cube_position"]
        inputs["cube_initial_orientation"] = first_stage4["data"]["cube_orientation"]
        inputs["cube_target_position"] = first_stage4["data"]["target_position"]
        T_grasp = get_relative_transform(first_stage4["data"]["tf"], "panda_hand", "world")
    

        # Count contacts after stage 3.
        for d in data_after_stage4:
            if d["data"]["contact"]:
                contact_count += 1

    if T_grasp is None:
        grasp_unsuccessful = True

    if not grasp_unsuccessful:
        T_grasp_inv = np.linalg.inv(T_grasp)

        # --- Transform cube initial pose into the grasp frame ---
        inputs["cube_initial_rel_position"], inputs["cube_initial_rel_orientation"] = transform_pose_to_frame(inputs["cube_initial_position"],
                                                                                                            inputs["cube_initial_orientation"],
                                                                                                            T_grasp_inv)
        
        
    


    ###############################
    ####### Outputs
    ###############################
    outputs = {
            "cube_final_position": None,
            "cube_final_orientation": None,
            "cube_final_surface": None,
            "position_difference": None,
            "orientation_difference": None,
            "shift_position": None,
            "shift_orientation": None,
            "cube_final_rel_position": None,        # Relative to the gripper frame when the cube just reached it's final pose.
            "cube_final_rel_orientation": None,     # Relative to the gripper frame when the cube just reached it's final pose.
            "contacts": contact_count,
            "grasp_unsuccessful": grasp_unsuccessful,
            "bad": False
    }
    

    if not grasp_unsuccessful: 

        delta_pose = None
        final_pose = None
        stage7 = [d for d in isaac_sim_data if d["data"]["stage"] == 7]
        if stage7:
            delta_pose = min(stage7, key=lambda x: x["current_time"])
            for i in range(len(isaac_sim_data)-1):
                if isaac_sim_data[i]["current_time"] >= delta_pose["current_time"]:
                    if isaac_sim_data[i]["data"]["cube_in_ground"] and isaac_sim_data[i+1]["data"]["cube_in_ground"]:
                        final_pose = isaac_sim_data[i]

                    # if (delta_pose is not None) and np.allclose(np.array(isaac_sim_data[i]["data"]["cube_position"]), 
                    #                np.array(isaac_sim_data[i+2]["data"]["cube_position"]), 
                    #                atol=1e-6):
                    #     final_pose = isaac_sim_data[i]
                    #     break
                    
        if final_pose is None:
            final_pose = isaac_sim_data[-1]
            outputs["bad"] = True


        
        T_release = get_relative_transform(final_pose["data"]["tf"], "panda_hand", "world")
        T_release_inv = np.linalg.inv(T_release)
        inputs["cube_target_orientation"] = compute_feasible_cube_pose(delta_pose["data"]).tolist()
        inputs["cube_target_rel_position"], inputs["cube_target_rel_orientation"] = transform_pose_to_frame(inputs["cube_target_position"],
                                                                                                            inputs["cube_target_orientation"],
                                                                                                            T_release_inv)



        # --- Cube final pose (world frame) ---
        outputs["cube_final_position"] = final_pose["data"]["cube_position"]
        outputs["cube_final_orientation"] = final_pose["data"]["cube_orientation"]
        outputs["cube_final_surface"] = final_pose["data"]["surface"]

        # Compute the differences (in world frame) between the final cube pose and the target pose.
        outputs["position_difference"], outputs["orientation_difference"] = pose_difference(
            outputs["cube_final_position"], outputs["cube_final_orientation"],
            inputs["cube_target_position"], inputs["cube_target_orientation"]
        )

        # Determine the delta pose from when the cube has settled (after stage 7).
        
        if delta_pose is not None:
            outputs["shift_position"], outputs["shift_orientation"] = pose_difference(
                outputs["cube_final_position"], outputs["cube_final_orientation"],
                delta_pose["data"]["cube_position"], delta_pose["data"]["cube_orientation"],
                exclude_z=True
            )


        # Transform cube final pose into the grasp frame.
        outputs["cube_final_rel_position"], outputs["cube_final_rel_orientation"] = transform_pose_to_frame(outputs["cube_final_position"],
                                                                                                            outputs["cube_final_orientation"],
                                                                                                            T_release_inv)
    
    # print(f"grasp position:{inputs['grasp_position'].__class__.__name__}")
    # print(f"grasp orientation:{inputs['grasp_orientation'].__class__.__name__}")
    # print(f"cube initial position:{inputs['cube_initial_position'].__class__.__name__}")
    # print(f"cube initial orientation:{inputs['cube_initial_orientation'].__class__.__name__}")
    # print(f"cube target position:{inputs['cube_target_position'].__class__.__name__}")
    # print(f"cube target orientation:{inputs['cube_target_orientation'].__class__.__name__}")
    # print(f"cube initial rel position:{inputs['cube_initial_rel_position'].__class__.__name__}")
    # print(f"cube initial rel orientation:{inputs['cube_initial_rel_orientation'].__class__.__name__}")
    # print(f"cube target rel position:{inputs['cube_target_rel_position'].__class__.__name__}")
    # print(f"cube target rel orientation:{inputs['cube_target_rel_orientation'].__class__.__name__}")
    # print(f"cube final position:{outputs['cube_final_position'].__class__.__name__}")
    # print(f"cube final orientation:{outputs['cube_final_orientation'].__class__.__name__}")
    # print(f"cube final surface:{outputs['cube_final_surface'].__class__.__name__}")
    # print(f"position difference:{outputs['position_difference'].__class__.__name__}")
    # print(f"orientation difference:{outputs['orientation_difference'].__class__.__name__}")
    # print(f"shift position:{outputs['shift_position'].__class__.__name__}")
    # print(f"shift orientation:{outputs['shift_orientation'].__class__.__name__}")
    # print(f"cube final rel position:{outputs['cube_final_rel_position'].__class__.__name__}")
    # print(f"cube final rel orientation:{outputs['cube_final_rel_orientation'].__class__.__name__}")
    # print(f"contacts:{outputs['contacts'].__class__.__name__}")
    # print(f"grasp unsuccessful:{outputs['grasp_unsuccessful'].__class__.__name__}")
    # print(f"bad:{outputs['bad'].__class__.__name__}")
 
    return {
        "file_path": file_path,
        "inputs": inputs,
        "outputs": outputs
    }




def process_folder(root_folder, output_file):
    """
    Process all JSON files in a folder and its subfolders, and save the extracted data.
    """
    results = []
    file_list = []

    # Collect all JSON file paths
    for subdir, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith(".json") and file_name != "Grasping.json":
                file_list.append(os.path.join(subdir, file_name))

    total_files = len(file_list)
    print(f"Total files to process: {total_files}")

    # Process each file while printing progress
    for idx, file_path in enumerate(file_list):
        files_left = total_files - idx - 1
        print(f"Processing file {idx+1}/{total_files}. Files left: {files_left}")
        try:
            data = process_file(file_path)
            results.append(data)
            print(f"Successfully processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            if 'quat' in str(e) or 'Eigenvalues' in str(e):
                os.remove(file_path)
                print(f"Removed {file_path}")

    # Save results to the output file (JSON format)
    with open(output_file, "w") as out_file:
        json.dump(results, out_file, indent=4)
    print(f"Processed data saved to {output_file}")
    
# Usage Example

# These are the "base" orientations in Euler angles [roll, pitch, yaw] (degrees) you said:
LOCAL_FACE_AXES = {
    "marker_top":  np.array([  0.0,   0.0,   0.0]),     # top face up  z diff
    "marker_bottom":  np.array([-180.0, 0.0,   0.0]),     # bottom face up z diff
    "marker_left":  np.array([ 90.0,  0.0,   0.0]),     # left face up y diff
    "marker_right":  np.array([-90.0,  0.0,   0.0]),     # right face up y diff
    "marker_front":  np.array([ 90.0,  0.0,  90.0]),     # front face up y diff
    "marker_back":  np.array([ 90.0,  0.0, -90.0]),     # back face up y diff
}

def compute_feasible_cube_pose(data):
    # ------------------------------------------------------------
    # (A) Extract relevant data
    # ------------------------------------------------------------
    cube_quat_current_wxyz = data["cube_orientation"]   # [w,x,y,z]
    current_marker = data["surface"]
    cube_quat_current_xyzw = convert_wxyz_to_xyzw(cube_quat_current_wxyz)

    # r_cube, p_cube, y_cube = tft.euler_from_quaternion(cube_quat_current_xyzw, axes='sxyz')


    best_q_base_xyzw = None
    for key, euler_deg in LOCAL_FACE_AXES.items():
        if key == current_marker:
            # 1) Convert the euler_deg -> radians -> quaternion base
            r_base_rad = math.radians(euler_deg[0])
            p_base_rad = math.radians(euler_deg[1])
            y_base_rad = math.radians(euler_deg[2])
            best_q_base_xyzw = tft.quaternion_from_euler(r_base_rad, p_base_rad, y_base_rad, axes='sxyz')
        
    # print(f"Your best key is: {current_marker}")
    # ------------------------------------------------------------
    # (E) Compute orientation difference: q_diff = q_pred * inv(q_base)
    #     Then convert to Euler, keep only one axis (e.g. pitch), zero out the others
    # ------------------------------------------------------------
    # By definition, if q_pred = q_diff * q_base, then q_diff = q_pred * q_base^-1
    q_base_inv = tft.quaternion_inverse(best_q_base_xyzw)
    q_diff_xyzw = tft.quaternion_multiply(cube_quat_current_xyzw, q_base_inv)

    # Convert that difference to Euler angles
    r_diff, p_diff, y_diff = tft.euler_from_quaternion(q_diff_xyzw, axes='sxyz')
    r_diff_deg = math.degrees(r_diff)
    p_diff_deg = math.degrees(p_diff)
    y_diff_deg = math.degrees(y_diff)

    # Suppose we only preserve pitch difference, zero out roll & yaw
    # (You can choose whichever axis you want to keep or partially keep)
    if current_marker in ["marker_top", "marker_bottom"]:
        # Preserve yaw -> zero out roll & pitch
        r_diff_deg_mod = 0.0
        p_diff_deg_mod = 0.0
        y_diff_deg_mod = y_diff_deg
    else:
        # Preserve pitch -> zero out roll & yaw
        r_diff_deg_mod = 0.0
        p_diff_deg_mod = p_diff_deg
        y_diff_deg_mod = 0.0

    rpy_final = LOCAL_FACE_AXES[current_marker] + np.array([r_diff_deg_mod, p_diff_deg_mod, y_diff_deg_mod])
    # print(f"Your final rpy is: {rpy_final}")
    # print(f"Your final rpy is: {rpy_final}")
    cube_quat = euler2quat(math.radians(rpy_final[0]), math.radians(rpy_final[1]), math.radians(rpy_final[2]), axes='sxyz') # [w,x,y,z]
    # print(f"ANd your final quat is: {cube_quat}")
    return cube_quat


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def build_transform(pos_xyz, quat_xyzw):
    """
    Build 4x4 from position [x,y,z] and quaternion [x,y,z,w].
    """
    T = tft.quaternion_matrix(quat_xyzw)
    T[:3, 3] = pos_xyz
    return T

def transform_to_pose(T):
    """
    Convert 4x4 -> (position, orientation [w,x,y,z])
    """
    pos = T[:3, 3].tolist()
    q_xyzw = tft.quaternion_from_matrix(T)  # [x,y,z,w]
    return pos, [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def quaternion_distance(q1_xyzw, q2_xyzw):
    """
    Simple measure: 1 - |dot(q1,q2)| for unit quaternions q1,q2 in [x,y,z,w] form.
    Ranges [0..2].
    """
    dot_val = abs(np.dot(q1_xyzw, q2_xyzw))
    return 1.0 - dot_val



if __name__ == "__main__":
    # process_folder("/home/chris/Chris/placement_ws/src/random_data", "/home/chris/Chris/placement_ws/src/placement_quality/grasp_placement/learning_models/data_intergration.json")
    # x = process_file("/home/chris/Chris/placement_ws/src/random_data/run_20250215_172420/Grasping_3/Placement_0_True.json")
    # reformat_json("/home/chris/Chris/placement_ws/src/random_data/Grasping_159/Placement_68_False.json")
    # rough_analysis("/home/chris/Chris/placement_ws/src/data/processed_data/data.json")
    # count_files_in_subfolders("/home/chris/Chris/placement_ws/src/data/random_data")
    # data_analysis("/home/chris/Chris/placement_ws/src/data/processed_data/data.json")
    # data_split()
    count_files_in_subfolders("/home/chris/Chris/placement_ws/src/data/YCB_data")