# Current data: {'grasp_pose': [0.20083635886413062, -0.26032669770493244, 0.2615313394317913, 0.11673269204954942, -0.12372194707573601, 0.9757111217871692, -0.13803682566430378], 'initial_object_pose': [0.14574891552329064, 0.6975172758102417, -0.6974669098854065, -0.11621776968240738, 0.11620249599218369], 'final_object_pose': [0.12549962885677815, 2.3502393560193013e-06, 0.3960959315299988, 0.9182091355323792, -1.1008057754224865e-07], 'angular_distance': 134.9639092750974, 'joint_distance': 4.173351774662387, 'manipulability': 0.31515759229660034, 'surface_up_initial': 'y_down', 'surface_up_final': 'z_down', 'surface_transition_type': 'adjacent', 'success_label': 1.0, 'collision_label': 1.0, 'ik_success': True, 'difficulty_score': -0.026497652245241088}

import json
import pandas as pd
import torch
import numpy as np
import copy
import os
import open3d as o3d
from typing import Dict, List, Any, Optional
import sys

# Add the model training directory to the path to import the model
sys.path.append('/home/chris/Chris/placement_ws/src/placement_quality/path_simulation/model_training')
from placement_quality.path_simulation.model_training.model import GraspObjectFeasibilityNet, PointNetEncoder
from placement_quality.path_simulation.model_training.train import farthest_point_sample, index_points

def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load the test data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def load_experiment_results(file_path: str) -> List[Dict[str, Any]]:
    """Load experiment results from JSONL file."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results

def combine_data(test_data: List[Dict[str, Any]], 
                experiment_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine test data with experiment results based on index.
    The index in experiment results corresponds to the position/order in test data.
    
    Args:
        test_data: List of test data dictionaries
        experiment_results: List of experiment result dictionaries
    
    Returns:
        List of combined data dictionaries
    """
    combined_data = []
    
    for exp_result in experiment_results:
        index = exp_result.get('index')
        if index is None:
            print(f"Warning: Experiment result missing index: {exp_result}")
            continue
            
        # Check if index is within bounds of test data
        if index >= len(test_data):
            print(f"Warning: Index {index} is out of bounds for test data (max: {len(test_data)-1})")
            continue
            
        # Get the corresponding test data by position
        test_entry = test_data[index]
        
        # Combine the data
        combined_entry = {
            # Test data fields
            'index': index,
            'grasp_pose': test_entry.get('grasp_pose'),
            'initial_object_pose': test_entry.get('initial_object_pose'),
            'final_object_pose': test_entry.get('final_object_pose'),
            'angular_distance': test_entry.get('angular_distance'),
            'joint_distance': test_entry.get('joint_distance'),
            'manipulability': test_entry.get('manipulability'),
            'surface_up_initial': test_entry.get('surface_up_initial'),
            'surface_up_final': test_entry.get('surface_up_final'),
            'surface_transition_type': test_entry.get('surface_transition_type'),
            'success_label': test_entry.get('success_label'),
            'collision_label': test_entry.get('collision_label'),
            'ik_success': test_entry.get('ik_success'),
            'difficulty_score': test_entry.get('difficulty_score'),
            
            # Experiment result fields
            'grasp': exp_result.get('grasp'),
            'collision_counter': exp_result.get('collision_counter'),
            'reason': exp_result.get('reason'),
            'forced_completion': exp_result.get('forced_completion')
        }
        
        combined_data.append(combined_entry)
    
    return combined_data

def save_combined_data(combined_data: List[Dict[str, Any]], 
                      output_file: str) -> None:
    """Save combined data to JSON file."""
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=2)

def model_prediction(model, data, device):
    """Run model prediction on a single data point."""
    # Define constant values as in the dataset
    const_xy = torch.tensor([0.2, -0.3], dtype=torch.float32)
    
    # Preprocess the initial and final poses as done in the dataset
    initial_pose = torch.cat([const_xy, torch.tensor(data["initial_object_pose"], dtype=torch.float32)])
    final_pose = torch.cat([const_xy, torch.tensor(data["final_object_pose"], dtype=torch.float32)])
    
    raw_success, raw_collision = model(None, 
                                     torch.tensor(data["grasp_pose"], dtype=torch.float32).unsqueeze(0).to(device), 
                                     initial_pose.unsqueeze(0).to(device), 
                                     final_pose.unsqueeze(0).to(device))
    
    # Apply sigmoid to convert logits to probabilities
    pred_success = torch.sigmoid(raw_success)
    pred_collision = torch.sigmoid(raw_collision)
    
    # Extract scalar values from tensors
    pred_success_val = pred_success.item()
    pred_collision_val = pred_collision.item()
    
    # Calculate no-collision probability (1 - collision probability)
    pred_no_collision_val = 1 - pred_collision_val
    
    # Calculate the combined score using no-collision probability
    score = pred_success_val * pred_no_collision_val
    
    return score, pred_success_val, pred_no_collision_val

def load_pointcloud(pcd_path, target_points=1024):
    """Load and downsample point cloud."""
    if type(pcd_path) == str:
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
    else:
        points = pcd_path
    
    # Convert to torch tensor for FPS
    points_tensor = torch.from_numpy(points).float().unsqueeze(0)  # [1, N, 3]
    
    # Apply farthest point sampling to downsample
    if len(points) > target_points:
        fps_idx = farthest_point_sample(points_tensor, target_points)
        points_downsampled = index_points(points_tensor, fps_idx).squeeze(0).numpy()
        print(f"Downsampled point cloud from {len(points)} to {len(points_downsampled)} points")
        return points_downsampled
    
    return points

def setup_model(checkpoint_path, pcd_path, device):
    """Setup the model with point cloud features."""
    # Load point cloud
    object_pcd_np = load_pointcloud(pcd_path)
    object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
    print(f"Loaded point cloud with {object_pcd.shape[0]} points...")
    
    # Forward once through PointNetEncoder
    with torch.no_grad():
        pn = PointNetEncoder(global_feat_dim=256).to(device)
        static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
    print("Point cloud features extracted.")

    # Load the checkpoint 
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    raw_state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    
    # Strip any unwanted prefix
    prefix_to_strip = "_orig_mod."
    cleaned_state_dict = {}
    for key, tensor in raw_state_dict.items():
        if key.startswith(prefix_to_strip):
            new_key = key[len(prefix_to_strip):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = tensor
    
    # Initialize model without static feature yet
    model = GraspObjectFeasibilityNet(use_static_obj=True).to(device)
    # Register this feature with the model
    model.register_buffer('static_obj_feat', static_obj_feat)
    # Load the state dict but don't compute point cloud embedding yet
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    print("Model loaded from checkpoint.")
    
    return model

def add_model_predictions(combined_data: List[Dict[str, Any]], 
                         checkpoint_path: str,
                         pcd_path: str) -> List[Dict[str, Any]]:
    """Add model predictions to the combined data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running predictions on device: {device}")
    
    # Setup model
    model = setup_model(checkpoint_path, pcd_path, device)
    
    # Add predictions to each data point
    for i, data_point in enumerate(combined_data):
        try:
            score, pred_success_val, pred_collision_val = model_prediction(model, data_point, device)
            
            # Add prediction results to the data point
            data_point['model_score'] = score
            data_point['model_success_prob'] = pred_success_val
            data_point['model_no_collision_prob'] = pred_collision_val
            
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(combined_data)} samples")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            # Add default values if prediction fails
            data_point['model_score'] = 0.0
            data_point['model_success_prob'] = 0.0
            data_point['model_no_collision_prob'] = 0.0
    
    print("Model predictions added to all data points.")
    return combined_data

def analyze_combined_data(combined_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the combined data and return statistics."""
    if not combined_data:
        return {}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(combined_data)
    
    analysis = {
        'total_samples': len(combined_data),
        'successful_grasps': len(df[df['grasp'] == True]) if 'grasp' in df.columns else 0,
        'failed_grasps': len(df[df['grasp'] == False]) if 'grasp' in df.columns else 0,
        'forced_completions': len(df[df['forced_completion'] == True]) if 'forced_completion' in df.columns else 0,
        'avg_collision_counter': df['collision_counter'].mean() if 'collision_counter' in df.columns else 0,
        'avg_difficulty_score': df['difficulty_score'].mean() if 'difficulty_score' in df.columns else 0,
        'avg_angular_distance': df['angular_distance'].mean() if 'angular_distance' in df.columns else 0,
        'avg_joint_distance': df['joint_distance'].mean() if 'joint_distance' in df.columns else 0,
        'avg_manipulability': df['manipulability'].mean() if 'manipulability' in df.columns else 0,
    }
    
    # Add model prediction statistics if available
    if 'model_score' in df.columns:
        analysis['avg_model_score'] = df['model_score'].mean()
        analysis['avg_model_success_prob'] = df['model_success_prob'].mean()
        analysis['avg_model_no_collision_prob'] = df['model_no_collision_prob'].mean()
    
    # Analyze reasons for failures
    if 'reason' in df.columns:
        reason_counts = {}
        for reason in df['reason']:
            if isinstance(reason, dict):
                # Extract position and orientation errors
                pos_error = reason.get('position_error', 0)
                orient_error = reason.get('orientation_error_deg', 0)
                reason_counts['position_error'] = reason_counts.get('position_error', 0) + 1
                reason_counts['orientation_error'] = reason_counts.get('orientation_error', 0) + 1
            elif isinstance(reason, str):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        analysis['failure_reasons'] = reason_counts
    
    return analysis

def main():
    """Main function to extract and combine data."""
    # File paths
    test_data_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/combined_data/data_with_difficulty.json"
    experiment_results_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/experiments/experiment_results.jsonl"
    output_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/combined_data/combined_experiment_data.json"
    
    # Model paths
    checkpoint_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/models/model_20250626_220307/best_model_0_1045_pth"
    pcd_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/experiments/run_20250701_010620/pointcloud.pcd"
    
    print("Loading test data...")
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} test data samples")
    
    print("Loading experiment results...")
    experiment_results = load_experiment_results(experiment_results_path)
    print(f"Loaded {len(experiment_results)} experiment results")
    
    print("Combining data...")
    combined_data = combine_data(test_data, experiment_results)
    print(f"Combined {len(combined_data)} data samples")
    
    print("Adding model predictions...")
    combined_data = add_model_predictions(combined_data, checkpoint_path, pcd_path)
    
    print("Analyzing data...")
    analysis = analyze_combined_data(combined_data)
    print("Analysis results:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print(f"Saving combined data to {output_path}...")
    save_combined_data(combined_data, output_path)
    print("Data extraction complete!")
    
    # Show a sample of the combined data
    if combined_data:
        print("\nSample combined data:")
        sample = combined_data[0]
        for key, value in sample.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()