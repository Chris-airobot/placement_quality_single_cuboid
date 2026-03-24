import torch
import numpy as np
import copy
import os
import open3d as o3d
from model import GraspObjectFeasibilityNet, PointNetEncoder
from train import farthest_point_sample, index_points
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def model_prediction(model, data, device):
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
    
    # Get binary predictions based on threshold of 0.5
    pred_success_binary = pred_success > 0.5
    pred_collision_binary = pred_collision > 0.5  # True means "collision predicted"
    
    return pred_success_val, 1-pred_collision_val
    return pred_success_binary.item(), pred_collision_binary.item()

# Load point cloud
def load_pointcloud(pcd_path, target_points=1024):
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

def analyze_collisions(df):
    """Analyze collision patterns"""
    collision_data = df[df['grasp'] == True].copy()
    
    # Categorize collision levels
    def categorize_collisions(counter):
        if counter == 0:
            return 'No Collisions'
        elif counter <= 10:
            return 'Low Collisions (1-10)'
        elif counter <= 50:
            return 'Medium Collisions (11-50)'
        else:
            return 'High Collisions (>50)'
    
    collision_data['collision_category'] = collision_data['collision_counter'].apply(categorize_collisions)
    
    collision_summary = collision_data['collision_category'].value_counts()
    
    # Model prediction accuracy for collisions
    collision_data['model_predicted_no_collision'] = collision_data['model_no_collision_prob'] > 0.5
    collision_data['actual_no_collision'] = collision_data['collision_counter'] == 0
    
    collision_accuracy = {
        'predicted_no_collision': len(collision_data[collision_data['model_predicted_no_collision']]),
        'actual_no_collision': len(collision_data[collision_data['actual_no_collision']]),
        'correct_predictions': len(collision_data[
            collision_data['model_predicted_no_collision'] == collision_data['actual_no_collision']
        ]),
        'false_positives': len(collision_data[
            (collision_data['model_predicted_no_collision'] == True) & 
            (collision_data['actual_no_collision'] == False)
        ]),
        'false_negatives': len(collision_data[
            (collision_data['model_predicted_no_collision'] == False) & 
            (collision_data['actual_no_collision'] == True)
        ])
    }
    
    return {
        'summary': collision_summary,
        'accuracy': collision_accuracy,
        'data': collision_data
    }

def analyze_model_performance(df):
    """Analyze model prediction performance"""
    successful_grasps = df[df['grasp'] == True].copy()
    
    # Success prediction accuracy
    successful_grasps['model_predicted_success'] = successful_grasps['model_success_prob'] > 0.5
    successful_grasps['actual_success'] = successful_grasps['reason'].apply(
        lambda x: isinstance(x, dict)  # If reason is dict, it's a success with pose difference
    )
    
    success_accuracy = {
        'predicted_success': len(successful_grasps[successful_grasps['model_predicted_success']]),
        'actual_success': len(successful_grasps[successful_grasps['actual_success']]),
        'correct_predictions': len(successful_grasps[
            successful_grasps['model_predicted_success'] == successful_grasps['actual_success']
        ]),
        'accuracy_percentage': len(successful_grasps[
            successful_grasps['model_predicted_success'] == successful_grasps['actual_success']
        ]) / len(successful_grasps) * 100
    }
    
    return {
        'success_accuracy': success_accuracy,
        'data': successful_grasps
    }

def analyze_pose_accuracy(df):
    """Analyze pose accuracy for successful grasps, using min/max normalization and outlier removal."""
    successful_grasps = df[df['grasp'] == True].copy()
    
    # Extract pose errors for successful grasps
    pose_errors = []
    for _, row in successful_grasps.iterrows():
        if isinstance(row['reason'], dict):
            position_error = row['reason']['position_error']
            orientation_error_deg = row['reason']['orientation_error_deg']
            pose_errors.append({
                'position_error': position_error,
                'orientation_error_deg': orientation_error_deg,
                'model_score': row['model_score'],
                'model_no_collision_prob': row['model_no_collision_prob'],
                'collision_counter': row['collision_counter'],
                'difficulty_score': row.get('difficulty_score', 0)
            })
    pose_df = pd.DataFrame(pose_errors)
    if len(pose_df) == 0:
        return {'data': pose_df}

    # Min/max for normalization
    pos_min, pos_max = pose_df['position_error'].min(), pose_df['position_error'].max()
    ori_min, ori_max = pose_df['orientation_error_deg'].min(), pose_df['orientation_error_deg'].max()

    # Avoid division by zero
    pos_range = pos_max - pos_min if pos_max > pos_min else 1.0
    ori_range = ori_max - ori_min if ori_max > ori_min else 1.0

    pose_df['normalized_position_error'] = (pose_df['position_error'] - pos_min) / pos_range
    pose_df['normalized_orientation_error'] = (pose_df['orientation_error_deg'] - ori_min) / ori_range

    # Combine using Euclidean norm
    pose_df['pose_difference'] = np.sqrt(
        pose_df['normalized_position_error']**2 + pose_df['normalized_orientation_error']**2
    )

    # Remove outliers in pose_difference (keep 1st to 99th percentile)
    lower = pose_df['pose_difference'].quantile(0.01)
    upper = pose_df['pose_difference'].quantile(0.99)
    pose_df = pose_df[(pose_df['pose_difference'] >= lower) & (pose_df['pose_difference'] <= upper)]

    return {
        'position_error_stats': pose_df['position_error'].describe(),
        'orientation_error_stats': pose_df['orientation_error_deg'].describe(),
        'pose_difference_stats': pose_df['pose_difference'].describe(),
        'data': pose_df
    }

def compare_predictions_vs_reality(df):
    """Compare model predictions with actual execution results"""
    successful_grasps = df[df['grasp'] == True].copy()
    
    # Create comparison categories
    def categorize_prediction_accuracy(row):
        model_score = row['model_score']
        collision_counter = row['collision_counter']
        
        if isinstance(row['reason'], dict):  # Successful execution
            if collision_counter == 0:
                if model_score > 0.8:
                    return 'High Prediction, No Collisions'
                elif model_score > 0.5:
                    return 'Medium Prediction, No Collisions'
                else:
                    return 'Low Prediction, No Collisions'
            else:
                if model_score > 0.8:
                    return 'High Prediction, Had Collisions'
                elif model_score > 0.5:
                    return 'Medium Prediction, Had Collisions'
                else:
                    return 'Low Prediction, Had Collisions'
        else:  # Failed execution
            if model_score > 0.8:
                return 'High Prediction, Failed'
            elif model_score > 0.5:
                return 'Medium Prediction, Failed'
            else:
                return 'Low Prediction, Failed'
    
    successful_grasps['prediction_category'] = successful_grasps.apply(categorize_prediction_accuracy, axis=1)
    
    return {
        'categories': successful_grasps['prediction_category'].value_counts(),
        'data': successful_grasps
    }

def create_visualizations(df, analysis):
    """Generate and save two plots: box plot of pose difference by model score, and model score histogram by collision level."""
    pose_data = analysis['pose_accuracy']['data']
    if len(pose_data) == 0:
        print("No pose data to plot.")
        return

    # Filter out outliers in pose_difference (keep 1st to 99th percentile)
    lower = pose_data['pose_difference'].quantile(0.01)
    upper = pose_data['pose_difference'].quantile(0.99)
    filtered = pose_data[(pose_data['pose_difference'] >= lower) & (pose_data['pose_difference'] <= upper)]

    # --------- 1. Box Plot: Pose Difference by Model Score Bin ---------
    filtered['score_bin'] = pd.cut(filtered['model_score'], bins=np.linspace(0, 1, 11), include_lowest=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='score_bin', y='pose_difference', data=filtered, showfliers=False, color='lightcoral')
    plt.xlabel('Model Prediction Score Bin')
    plt.ylabel('Normalized Pose Difference')
    plt.title('Pose Difference Distribution by Model Prediction Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('pose_difference_by_model_score.png')
    plt.close()

    # --------- 2. Histogram: Model Score Distribution by Collision Level ---------
    # Add collision category to pose data
    collision_categories = ['No Collisions', 'Low Collisions (1-10)', 'Medium Collisions (11-50)', 'High Collisions (>50)']
    filtered['collision_category'] = filtered['collision_counter'].apply(
        lambda x: 'No Collisions' if x == 0 else
                  'Low Collisions (1-10)' if x <= 10 else
                  'Medium Collisions (11-50)' if x <= 50 else
                  'High Collisions (>50)'
    )

    plt.figure(figsize=(10, 6))
    for category in collision_categories:
        if category in filtered['collision_category'].values:
            category_data = filtered[filtered['collision_category'] == category]['model_score']
            plt.hist(category_data, alpha=0.7, label=category, bins=20)
    plt.xlabel('Model Score')
    plt.ylabel('Frequency')
    plt.title('Model Score Distribution by Collision Level')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_score_by_collision_level.png')
    plt.close()

    print("Plots saved as 'pose_difference_by_model_score.png' and 'model_score_by_collision_level.png'")

def load_and_analyze_data(file_path, model, device):
    """Load and analyze the experiment data with model predictions"""
    
    # Load data - handle both JSONL and regular JSON formats
    try:
        # First try to load as regular JSON (array of objects)
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded data as regular JSON array with {len(data)} items")
    except json.JSONDecodeError:
        # If that fails, try JSONL format (one JSON object per line)
        try:
            with open(file_path, 'r') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            print(f"Loaded data as JSONL with {len(data)} items")
        except json.JSONDecodeError as e:
            print(f"Error loading data: {e}")
            print("Please check if the file is in valid JSON or JSONL format")
            return None, None
    
    # Add model predictions to each data point
    print("🔍 Computing model predictions for all data points...")
    for i, data_point in enumerate(data):
        if i % 100 == 0:
            print(f"   Processing data point {i}/{len(data)}")
        
        pred_success_val, pred_collision_val = model_prediction(model, data_point, device)
        score = pred_success_val * pred_collision_val
        
        # Add predictions to the data
        data_point['model_score'] = score
        data_point['model_success_prob'] = pred_success_val
        data_point['model_no_collision_prob'] = pred_collision_val
    
    df = pd.DataFrame(data)
    
    # Create analysis results
    analysis = {
        'total_experiments': len(df),
        'successful_grasps': len(df[df['grasp'] == True]),
        'failed_grasps': len(df[df['grasp'] == False]),
        'success_rate': len(df[df['grasp'] == True]) / len(df) * 100,
        'collision_analysis': analyze_collisions(df),
        'model_performance': analyze_model_performance(df),
        'pose_accuracy': analyze_pose_accuracy(df),
        'prediction_vs_reality': compare_predictions_vs_reality(df)
    }
    
    return df, analysis

def create_difficulty_performance_table(df):
    """Create a comprehensive table using existing difficulty_score with model predictions and actual results"""
    
    # Create difficulty bins based on the existing difficulty_score
    difficulty_bins = pd.cut(
        df['difficulty_score'], 
        bins=5, 
        labels=['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
    )
    df['difficulty_level'] = difficulty_bins
    
    # Calculate performance metrics for each difficulty level
    performance_table = []
    
    for level in df['difficulty_level'].unique():
        if pd.isna(level):
            continue
            
        subset = df[df['difficulty_level'] == level]
        successful = subset[subset['grasp'] == True]
        
        # Basic metrics
        total_experiments = len(subset)
        success_rate = len(successful) / total_experiments * 100 if total_experiments > 0 else 0
        avg_model_score = subset['model_score'].mean()
        avg_difficulty = subset['difficulty_score'].mean()
        
        # Model prediction accuracy
        subset['model_predicted_success'] = subset['model_success_prob'] > 0.5
        prediction_accuracy = (subset['model_predicted_success'] == subset['grasp']).mean() * 100
        
        # Pose accuracy for successful grasps
        pose_errors = []
        for _, row in successful.iterrows():
            if isinstance(row['reason'], dict):
                pose_errors.append(row['reason']['position_error'])
        
        avg_position_error = np.mean(pose_errors) if pose_errors else 0
        
        # Collision metrics
        no_collision_rate = (subset['collision_counter'] == 0).mean() * 100
        avg_collisions = subset['collision_counter'].mean()
        
        # Difficulty range
        min_diff = subset['difficulty_score'].min()
        max_diff = subset['difficulty_score'].max()
        
        performance_table.append({
            'Difficulty_Level': level,
            'Difficulty_Range': f"{min_diff:.3f}-{max_diff:.3f}",
            'Avg_Difficulty': round(avg_difficulty, 3),
            'Total_Experiments': total_experiments,
            'Success_Rate_%': round(success_rate, 1),
            'Avg_Model_Score': round(avg_model_score, 3),
            'Prediction_Accuracy_%': round(prediction_accuracy, 1),
            'Avg_Position_Error_m': round(avg_position_error, 4),
            'No_Collision_Rate_%': round(no_collision_rate, 1),
            'Avg_Collisions': round(avg_collisions, 1)
        })
    
    return pd.DataFrame(performance_table)

def print_difficulty_performance_table(df):
    """Print a formatted difficulty-performance table"""
    table_df = create_difficulty_performance_table(df)
    
    print("\n" + "="*140)
    print("DIFFICULTY SCORE vs MODEL PERFORMANCE & ACTUAL RESULTS")
    print("="*140)
    print(f"{'Difficulty':<12} {'Range':<15} {'Avg_Diff':<8} {'Total':<6} {'Success%':<8} {'Model':<6} {'Pred%':<6} {'PosErr':<7} {'NoCol%':<7} {'AvgCol':<6}")
    print("-" * 140)
    
    for _, row in table_df.iterrows():
        print(f"{row['Difficulty_Level']:<12} {row['Difficulty_Range']:<15} {row['Avg_Difficulty']:<8.3f} {row['Total_Experiments']:<6} "
              f"{row['Success_Rate_%']:<8.1f} {row['Avg_Model_Score']:<6.3f} {row['Prediction_Accuracy_%']:<6.1f} "
              f"{row['Avg_Position_Error_m']:<7.4f} {row['No_Collision_Rate_%']:<7.1f} {row['Avg_Collisions']:<6.1f}")
    
    print("\n" + "="*140)
    print("Legend:")
    print("Difficulty: Based on angular_distance, joint_distance, manipulability, and surface_transition_type deviations")
    print("Range: Min-max values of difficulty_score for this level")
    print("Avg_Diff: Average difficulty_score for this level")
    print("Total: Number of experiments in this difficulty level")
    print("Success%: Percentage of successful grasps")
    print("Model: Average model prediction score")
    print("Pred%: Model prediction accuracy percentage")
    print("PosErr: Average position error in meters")
    print("NoCol%: Percentage of experiments with no collisions")
    print("AvgCol: Average number of collisions")
    print("="*140)

def analyze_model_vs_difficulty_correlation(df):
    """Analyze how well the model performs across different difficulty levels"""
    
    # Create difficulty bins
    df['difficulty_bin'] = pd.cut(df['difficulty_score'], bins=10)
    
    # Calculate metrics for each bin
    bin_analysis = df.groupby('difficulty_bin').agg({
        'difficulty_score': 'mean',
        'grasp': 'mean',
        'model_score': 'mean',
        'model_success_prob': 'mean',
        'model_no_collision_prob': 'mean',
        'collision_counter': 'mean'
    }).reset_index()
    
    # Calculate correlations
    correlations = {
        'difficulty_vs_success': df['difficulty_score'].corr(df['grasp']),
        'difficulty_vs_model_score': df['difficulty_score'].corr(df['model_score']),
        'difficulty_vs_collisions': df['difficulty_score'].corr(df['collision_counter']),
        'model_score_vs_success': df['model_score'].corr(df['grasp']),
        'model_score_vs_collisions': df['model_score'].corr(df['collision_counter'])
    }
    
    return bin_analysis, correlations

def create_difficulty_analysis_plots(df):
    """Create plots showing the relationship between difficulty and performance"""
    
    # Create difficulty bins for plotting
    df['difficulty_bin'] = pd.cut(df['difficulty_score'], bins=10)
    
    # Calculate metrics for each bin
    bin_metrics = df.groupby('difficulty_bin').agg({
        'difficulty_score': 'mean',
        'grasp': 'mean',
        'model_score': 'mean',
        'collision_counter': 'mean',
        'model_success_prob': 'mean',
        'model_no_collision_prob': 'mean'
    }).reset_index()
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Success rate vs difficulty
    ax1.plot(bin_metrics['difficulty_score'], bin_metrics['grasp'] * 100, 'o-', color='green', linewidth=2, markersize=8)
    ax1.set_xlabel('Difficulty Score')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Actual Success Rate vs Difficulty')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. Model score vs difficulty
    ax2.plot(bin_metrics['difficulty_score'], bin_metrics['model_score'], 'o-', color='blue', linewidth=2, markersize=8)
    ax2.set_xlabel('Difficulty Score')
    ax2.set_ylabel('Average Model Score')
    ax2.set_title('Model Confidence vs Difficulty')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. Collisions vs difficulty
    ax3.plot(bin_metrics['difficulty_score'], bin_metrics['collision_counter'], 'o-', color='red', linewidth=2, markersize=8)
    ax3.set_xlabel('Difficulty Score')
    ax3.set_ylabel('Average Collisions')
    ax3.set_title('Collisions vs Difficulty')
    ax3.grid(True, alpha=0.3)
    
    # 4. Model prediction accuracy scatter
    successful = df[df['grasp'] == True]
    failed = df[df['grasp'] == False]
    
    ax4.scatter(successful['difficulty_score'], successful['model_score'], 
               alpha=0.6, color='green', label='Successful', s=30)
    ax4.scatter(failed['difficulty_score'], failed['model_score'], 
               alpha=0.6, color='red', label='Failed', s=30)
    ax4.set_xlabel('Difficulty Score')
    ax4.set_ylabel('Model Score')
    ax4.set_title('Model Score vs Difficulty (by Outcome)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('difficulty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Difficulty analysis plots saved as 'difficulty_analysis.png'")

def create_detailed_difficulty_breakdown(df):
    """Create a detailed breakdown showing model performance for different difficulty ranges"""
    
    # Define custom difficulty ranges based on your data distribution
    difficulty_ranges = [
        (-float('inf'), -0.5, 'Very Easy'),
        (-0.5, 0.0, 'Easy'),
        (0.0, 0.5, 'Medium'),
        (0.5, 1.0, 'Hard'),
        (1.0, float('inf'), 'Very Hard')
    ]
    
    breakdown_data = []
    
    for min_val, max_val, label in difficulty_ranges:
        if min_val == -float('inf'):
            subset = df[df['difficulty_score'] <= max_val]
        elif max_val == float('inf'):
            subset = df[df['difficulty_score'] > min_val]
        else:
            subset = df[(df['difficulty_score'] > min_val) & (df['difficulty_score'] <= max_val)]
        
        if len(subset) == 0:
            continue
        
        successful = subset[subset['grasp'] == True]
        
        # Calculate metrics
        success_rate = len(successful) / len(subset) * 100
        avg_model_score = subset['model_score'].mean()
        
        # Model prediction accuracy
        subset['model_predicted_success'] = subset['model_success_prob'] > 0.5
        prediction_accuracy = (subset['model_predicted_success'] == subset['grasp']).mean() * 100
        
        # High confidence predictions (model_score > 0.8)
        high_conf = subset[subset['model_score'] > 0.8]
        high_conf_success_rate = len(high_conf[high_conf['grasp'] == True]) / len(high_conf) * 100 if len(high_conf) > 0 else 0
        
        # Low confidence predictions (model_score < 0.3)
        low_conf = subset[subset['model_score'] < 0.3]
        low_conf_success_rate = len(low_conf[low_conf['grasp'] == True]) / len(low_conf) * 100 if len(low_conf) > 0 else 0
        
        breakdown_data.append({
            'Difficulty_Range': label,
            'Score_Range': f"{min_val:.1f}-{max_val:.1f}",
            'Total_Experiments': len(subset),
            'Success_Rate_%': round(success_rate, 1),
            'Avg_Model_Score': round(avg_model_score, 3),
            'Prediction_Accuracy_%': round(prediction_accuracy, 1),
            'High_Conf_Success_%': round(high_conf_success_rate, 1),
            'Low_Conf_Success_%': round(low_conf_success_rate, 1),
            'High_Conf_Count': len(high_conf),
            'Low_Conf_Count': len(low_conf)
        })
    
    return pd.DataFrame(breakdown_data)

def print_detailed_breakdown(df):
    """Print detailed difficulty breakdown"""
    breakdown_df = create_detailed_difficulty_breakdown(df)
    
    print("\n" + "="*120)
    print("DETAILED DIFFICULTY BREAKDOWN WITH CONFIDENCE ANALYSIS")
    print("="*120)
    print(f"{'Range':<12} {'Score':<12} {'Total':<6} {'Success%':<8} {'Model':<6} {'Pred%':<6} {'HighConf%':<9} {'LowConf%':<9} {'High#':<6} {'Low#':<6}")
    print("-" * 120)
    
    for _, row in breakdown_df.iterrows():
        print(f"{row['Difficulty_Range']:<12} {row['Score_Range']:<12} {row['Total_Experiments']:<6} "
              f"{row['Success_Rate_%']:<8.1f} {row['Avg_Model_Score']:<6.3f} {row['Prediction_Accuracy_%']:<6.1f} "
              f"{row['High_Conf_Success_%']:<9.1f} {row['Low_Conf_Success_%']:<9.1f} "
              f"{row['High_Conf_Count']:<6} {row['Low_Conf_Count']:<6}")
    
    print("\n" + "="*120)
    print("Legend:")
    print("Range: Difficulty level based on your calculated difficulty_score")
    print("Score: Actual difficulty_score range")
    print("Total: Number of experiments in this range")
    print("Success%: Percentage of successful grasps")
    print("Model: Average model prediction score")
    print("Pred%: Model prediction accuracy percentage")
    print("HighConf%: Success rate for high confidence predictions (>0.8)")
    print("LowConf%: Success rate for low confidence predictions (<0.3)")
    print("High#: Number of high confidence predictions")
    print("Low#: Number of low confidence predictions")
    print("="*120)

def main(checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Load point cloud
    saved_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/experiments/run_20250701_010620/pointcloud.pcd"
    
    # Convert pointcloud to tensor
    object_pcd_np = load_pointcloud(saved_path)
    object_pcd = torch.tensor(object_pcd_np, dtype=torch.float32).to(device)
    print(f"Loaded point cloud with {object_pcd.shape[0]} points...")
    
    # Forward once through PointNetEncoder
    with torch.no_grad():
        pn = PointNetEncoder(global_feat_dim=256).to(device)
        static_obj_feat = pn(object_pcd.unsqueeze(0)).detach()   # [1,256]
    print("Done.\n")

    # Load the checkpoint 
    checkpoint_data = torch.load(checkpoint, map_location=device)
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
    print("Model loaded from checkpoint.\n")

    # Load and analyze data
    data_file_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/combined_data/combined_experiment_data.json"
    print("🔍 Loading and analyzing experiment data...")
    df, analysis = load_and_analyze_data(data_file_path, model, device)
    
    if df is None or analysis is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    print("📈 Creating visualizations...")
    create_visualizations(df, analysis)
    
    # Additional detailed analysis
    print("\n DETAILED BREAKDOWN BY PREDICTION CATEGORIES:")
    categories = analysis['prediction_vs_reality']['categories']
    for category, count in categories.items():
        percentage = count / analysis['total_experiments'] * 100
        print(f"   {category}: {count} experiments ({percentage:.1f}%)")
    
    # Show some example cases
    print("\n📋 EXAMPLE CASES:")
    successful_grasps = df[df['grasp'] == True]
    
    # High prediction, no collisions
    high_pred_no_collision = successful_grasps[
        (successful_grasps['model_score'] > 0.8) & 
        (successful_grasps['collision_counter'] == 0)
    ]
    if len(high_pred_no_collision) > 0:
        example = high_pred_no_collision.iloc[0]
        print(f"   ✅ High Prediction + No Collisions (Model Score: {example['model_score']:.3f})")
        if isinstance(example['reason'], dict):
            print(f"      Position Error: {example['reason']['position_error']:.4f}m, Orientation Error: {example['reason']['orientation_error_deg']:.1f}°")
    
    # High prediction, but had collisions
    high_pred_with_collision = successful_grasps[
        (successful_grasps['model_score'] > 0.8) & 
        (successful_grasps['collision_counter'] > 0)
    ]
    if len(high_pred_with_collision) > 0:
        example = high_pred_with_collision.iloc[0]
        print(f"   ⚠️  High Prediction + Had Collisions (Model Score: {example['model_score']:.3f}, Collisions: {example['collision_counter']})")
        if isinstance(example['reason'], dict):
            print(f"      Position Error: {example['reason']['position_error']:.4f}m, Orientation Error: {example['reason']['orientation_error_deg']:.1f}°")
    
    # Low prediction, but no collisions
    low_pred_no_collision = successful_grasps[
        (successful_grasps['model_score'] < 0.3) & 
        (successful_grasps['collision_counter'] == 0)
    ]
    if len(low_pred_no_collision) > 0:
        example = low_pred_no_collision.iloc[0]
        print(f"   ❓ Low Prediction + No Collisions (Model Score: {example['model_score']:.3f})")
        if isinstance(example['reason'], dict):
            print(f"      Position Error: {example['reason']['position_error']:.4f}m, Orientation Error: {example['reason']['orientation_error_deg']:.1f}°")

    # ADD THE DIFFICULTY ANALYSIS HERE - AFTER df IS DEFINED
    print("\n📊 Creating difficulty-performance table using existing difficulty_score...")
    print_difficulty_performance_table(df)

    print("\n📈 Creating detailed difficulty breakdown...")
    print_detailed_breakdown(df)

    print("\n🔗 Analyzing correlations...")
    bin_analysis, correlations = analyze_model_vs_difficulty_correlation(df)
    print("\nCorrelations:")
    for metric, corr in correlations.items():
        print(f"   {metric}: {corr:.3f}")

    print("\n📊 Creating difficulty analysis plots...")
    create_difficulty_analysis_plots(df)

if __name__ == "__main__":
    main(checkpoint="/home/chris/Chris/placement_ws/src/data/box_simulation/v2/models/model_20250626_220307/best_model_0_1045_pth")