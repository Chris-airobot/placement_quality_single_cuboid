import os
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
from tqdm import tqdm

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def analyze_class_distribution(data, output_dir):
    print("[1/6] Analyzing class distributions…")
    succ = [s["success_label"] for s in data]
    coll = [s["collision_label"] for s in data]
    scount = Counter(succ)
    ccount = Counter(coll)
    # Save JSON
    with open(os.path.join(output_dir, "label_counts.json"), "w") as f:
        json.dump({"success": dict(scount), "collision": dict(ccount)}, f, indent=2)
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    axes[0].bar(scount.keys(), scount.values(), color=['red','green'])
    axes[0].set_title("Success Label Distribution")
    axes[0].set_xticks([0,1]); axes[0].set_xticklabels(["Fail","Success"])
    axes[1].bar(ccount.keys(), ccount.values(), color=['green','red'])
    axes[1].set_title("Collision Label Distribution")
    axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["No Collision","Collision"])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "label_distribution.png"))
    plt.close(fig)
    print("    → Saved label_counts.json & label_distribution.png")

def analyze_pose_distributions(data, output_dir):
    print("[2/6] Analyzing position distributions…")
    init_z = np.array([s["initial_object_pose"][0] for s in data])
    final_pos = np.array([s["final_object_pose"][:3] for s in data])
    # Save JSON
    stats = {
        "init_z_mean": float(init_z.mean()), "init_z_std": float(init_z.std()),
        "final_x_mean": float(final_pos[:,0].mean()),
        "final_y_mean": float(final_pos[:,1].mean()),
        "final_z_mean": float(final_pos[:,2].mean()),
        "final_x_std":  float(final_pos[:,0].std()),
        "final_y_std":  float(final_pos[:,1].std()),
        "final_z_std":  float(final_pos[:,2].std()),
    }
    with open(os.path.join(output_dir, "position_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # Plot
    fig, axes = plt.subplots(2,1, figsize=(6,8))
    axes[0].hist(init_z, bins=50, color='blue', alpha=0.7)
    axes[0].set_title("Initial Z Distribution")
    axes[0].set_xlabel("Z")
    axes[1].hist(final_pos.flatten(), bins=50, color='orange', alpha=0.7)
    axes[1].set_title("Final XYZ Distribution (combined)")
    axes[1].set_xlabel("Value")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "position_distributions.png"))
    plt.close(fig)
    print("    → Saved position_stats.json & position_distributions.png")

def analyze_grasp_pose_variance(data, output_dir):
    print("[3/6] Analyzing grasp-pose orientation variance…")
    quats = np.array([s["grasp_pose"][3:] for s in data])
    norms = np.linalg.norm(quats, axis=1)
    # Save JSON
    stats = {
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
    }
    with open(os.path.join(output_dir, "grasp_quat_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # Plot histogram
    fig = plt.figure(figsize=(6,4))
    plt.hist(norms, bins=100, alpha=0.7)
    plt.title("Grasp Quaternion Norms")
    plt.savefig(os.path.join(output_dir, "grasp_quat_norms.png"))
    plt.close(fig)
    # PCA scatter
    pca = PCA(n_components=2).fit_transform(quats)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(pca[:,0], pca[:,1], alpha=0.2)
    plt.title("Grasp Orientation PCA")
    plt.savefig(os.path.join(output_dir, "grasp_quat_pca.png"))
    plt.close(fig)
    print("    → Saved grasp_quat_stats.json, grasp_quat_norms.png & grasp_quat_pca.png")

def analyze_orientation_change_vs_success(data, output_dir):
    print("[4/6] Computing orientation-change vs success…")
    angles = []
    labels = []
    for s in tqdm(data, desc="    Computing angles"):
        init = s["initial_object_pose"]
        r1 = R.from_quat([init[4], init[1], init[2], init[3]])
        # final always full
        r2 = R.from_quat([s["final_object_pose"][4], 
                          s["final_object_pose"][1], 
                          s["final_object_pose"][2], 
                          s["final_object_pose"][3]])
        angles.append((r1.inv() * r2).magnitude() * (180/np.pi))
        labels.append(s["success_label"])
    angles = np.array(angles); labels = np.array(labels)
    stats = {
        "mean_angle_success": float(angles[labels==1].mean()),
        "mean_angle_fail":    float(angles[labels==0].mean()),
    }
    with open(os.path.join(output_dir, "angle_change_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    # Plot
    fig = plt.figure(figsize=(8,5))
    plt.hist(angles[labels==1], bins=50, alpha=0.6, label="Success")
    plt.hist(angles[labels==0], bins=50, alpha=0.6, label="Fail")
    plt.legend(); plt.title("Orientation Change vs Success")
    plt.xlabel("Angle (°)")
    plt.savefig(os.path.join(output_dir, "angle_vs_success.png"))
    plt.close(fig)
    print("    → Saved angle_change_stats.json & angle_vs_success.png")

def analyze_surface_transitions(data, output_dir):
    print("[5/6] Analyzing surface-to-surface transitions…")
    trans = [s["surfaces"] for s in data]
    counter = Counter(trans)
    with open(os.path.join(output_dir, "surface_counts.json"), "w") as f:
        json.dump(dict(counter), f, indent=2)
    matrix = np.zeros((6,6), dtype=int)
    for k,v in counter.items():
        i,j = map(int, k.split("_"))
        matrix[i,j] = v
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(matrix, origin='lower')
    ax.set_xlabel("Final surface"); ax.set_ylabel("Initial surface")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "surface_transitions.png"))
    plt.close(fig)
    print("    → Saved surface_counts.json & surface_transitions.png")

def visualize_random_samples(data, output_dir, num=5):
    print("[6/6] Logging random samples…")
    path = os.path.join(output_dir, "random_samples.txt")
    with open(path, "w") as f:
        for idx in np.random.choice(len(data), num, replace=False):
            s = data[idx]
            f.write(f"Index {idx} | grasp: {s['grasp_pose']} | init: {s['initial_object_pose']} | final: {s['final_object_pose']}  | success: {s['success_label']} | collision: {s['collision_label']}\n")
    print("    → Saved random_samples.txt")

def write_report(output_dir):
    report = []
    report.append("=== Data Analysis Report ===\n")
    # Load JSON stats
    with open(os.path.join(output_dir, "label_counts.json")) as f:
        lc = json.load(f)
    report.append("Class Distribution:\n")
    for k,v in lc["success"].items():
        report.append(f"  Success={k}: {v}\n")
    for k,v in lc["collision"].items():
        report.append(f"  Collision={k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "position_stats.json")) as f:
        ps = json.load(f)
    report.append("Position Statistics:\n")
    for k,v in ps.items():
        report.append(f"  {k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "grasp_quat_stats.json")) as f:
        gs = json.load(f)
    report.append("Grasp Quaternion Norms:\n")
    for k,v in gs.items():
        report.append(f"  {k}: {v}\n")
    report.append("\n")
    with open(os.path.join(output_dir, "angle_change_stats.json")) as f:
        ang = json.load(f)
    report.append("Orientation Change Stats:\n")
    for k,v in ang.items():
        report.append(f"  {k}: {v}\n")
    report.append("\n")
    report.append("\nRandom Samples:\n")
    with open(os.path.join(output_dir, "random_samples.txt")) as f:
        report.extend(f.readlines())
    with open(os.path.join(output_dir, "analysis_report.txt"), "w") as f:
        f.writelines(report)
    print(f"Analysis report saved to {os.path.join(output_dir, 'analysis_report.txt')}")

def calculate_difficulty(new_case, reference_stats):
    """Calculate difficulty score for a new case"""
    
    # How far is angular_distance from average? (higher = harder)
    angular_deviation = (new_case["angular_distance"] - reference_stats['angular_avg']) / reference_stats['angular_std']
    
    # How far is joint_distance from average? (higher = harder)
    joint_deviation = (new_case["joint_distance"] - reference_stats['joint_avg']) / reference_stats['joint_std']
    
    # How far is manipulability from average? (lower = harder, so flip the sign)
    manip_deviation = -(new_case["manipulability"] - reference_stats['manip_avg']) / reference_stats['manip_std']
    
    # Surface transition difficulty (you can adjust these values)
    transition_scores = {
        'adjacent': 0.5,    # Medium difficulty
        'opposite': 1.0,    # Hardest
        'same': 0.0,        # Easiest
        # Add other types as needed
    }
    surface_score = transition_scores.get(new_case["surface_transition_type"], 0.5)
    
    # Combine them (normalize surface score to similar scale)
    difficulty = (angular_deviation + joint_deviation + manip_deviation + surface_score) / 4
    
    return difficulty

def analyze_difficulty_simple(data, output_dir):
    print("[7/7] Analyzing difficulty metrics...")
    
    # Step 1: Filter out cases with missing values and calculate reference statistics
    valid_data = []
    for sample in data:
        if (sample.get("angular_distance") is not None and 
            sample.get("joint_distance") is not None and 
            sample.get("manipulability") is not None):
            valid_data.append(sample)
    
    print(f"    → Found {len(valid_data)} valid samples out of {len(data)} total")
    
    if len(valid_data) == 0:
        print("    → No valid samples found! Check your data format.")
        return
    
    angular_distances = [s["angular_distance"] for s in valid_data]
    joint_distances = [s["joint_distance"] for s in valid_data]
    manipulabilities = [s["manipulability"] for s in valid_data]
    
    reference_stats = {
        'angular_avg': np.mean(angular_distances),
        'angular_std': np.std(angular_distances),
        'joint_avg': np.mean(joint_distances),
        'joint_std': np.std(joint_distances),
        'manip_avg': np.mean(manipulabilities),
        'manip_std': np.std(manipulabilities)
    }
    
    print(f"    → Reference stats calculated:")
    print(f"      Angular: avg={reference_stats['angular_avg']:.3f}, std={reference_stats['angular_std']:.3f}")
    print(f"      Joint: avg={reference_stats['joint_avg']:.3f}, std={reference_stats['joint_std']:.3f}")
    print(f"      Manip: avg={reference_stats['manip_avg']:.3f}, std={reference_stats['manip_std']:.3f}")
    
    # Step 2: Calculate difficulty for each valid case
    difficulties = []
    for sample in valid_data:
        difficulty = calculate_difficulty(sample, reference_stats)
        difficulties.append(difficulty)
    
    # Step 3: Save reference stats for future use
    with open(os.path.join(output_dir, "difficulty_reference_stats.json"), "w") as f:
        json.dump(reference_stats, f, indent=2)
    
    # Step 4: Add difficulty scores to valid data
    for i, sample in enumerate(valid_data):
        sample['difficulty_score'] = difficulties[i]
    
    # Step 5: Save updated data
    with open(os.path.join(output_dir, "data_with_difficulty.json"), "w") as f:
        json.dump(valid_data, f, indent=2)
    
    # Step 6: Analyze difficulty distribution
    difficulties = np.array(difficulties)
    stats = {
        "mean_difficulty": float(difficulties.mean()),
        "std_difficulty": float(difficulties.std()),
        "min_difficulty": float(difficulties.min()),
        "max_difficulty": float(difficulties.max()),
        "percentiles": {
            "25": float(np.percentile(difficulties, 25)),
            "50": float(np.percentile(difficulties, 50)),
            "75": float(np.percentile(difficulties, 75)),
            "90": float(np.percentile(difficulties, 90))
        }
    }
    
    with open(os.path.join(output_dir, "difficulty_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Step 7: Plot difficulty distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(difficulties, bins=50, alpha=0.7, color='blue')
    ax.axvline(difficulties.mean(), color='red', linestyle='--', label=f'Mean: {difficulties.mean():.3f}')
    ax.axvline(np.percentile(difficulties, 75), color='orange', linestyle='--', label=f'75th percentile: {np.percentile(difficulties, 75):.3f}')
    ax.set_xlabel('Difficulty Score')
    ax.set_ylabel('Count')
    ax.set_title('Difficulty Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "difficulty_distribution.png"))
    plt.close()
    
    print(f"    → Saved difficulty scores. Average difficulty: {difficulties.mean():.3f}")
    print(f"    → Reference stats saved for future use")
    print(f"    → Difficulty range: {difficulties.min():.3f} to {difficulties.max():.3f}")

def analyze_model_collision_prediction(data, output_dir):
    """
    Analyze model's collision prediction performance by finding optimal threshold
    and comparing model scores with actual collision data.
    """
    print("[7/7] Analyzing model collision prediction performance...")
    
    # Extract model scores and actual collision labels
    model_scores = []
    actual_collisions = []
    valid_indices = []
    
    for i, sample in enumerate(data):
        if "model_no_collision_prob" in sample and "collision_counter" in sample:
            model_scores.append(sample["model_no_collision_prob"])
            # Convert collision_counter to binary: > 0 means collision
            actual_collisions.append(1 if sample["collision_counter"] > 0 else 0)
            valid_indices.append(i)
    
    if len(model_scores) == 0:
        print("    → No valid samples found with model_no_collision_prob and collision_counter!")
        return
    
    model_scores = np.array(model_scores)
    actual_collisions = np.array(actual_collisions)
    
    print(f"    → Found {len(model_scores)} valid samples")
    print(f"    → Model score range: {model_scores.min():.4f} to {model_scores.max():.4f}")
    print(f"    → Actual collisions: {np.sum(actual_collisions)} out of {len(actual_collisions)} ({np.mean(actual_collisions)*100:.1f}%)")
    
    # Calculate ROC curve and find optimal threshold
    fpr, tpr, roc_thresholds = roc_curve(actual_collisions, model_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(actual_collisions, model_scores)
    pr_auc = auc(recall, precision)
    
    # Find optimal threshold using different methods
    # Method 1: Youden's J statistic (maximizes TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_youden = roc_thresholds[optimal_idx]
    
    # Method 2: Closest to (0,1) on ROC curve
    distances = np.sqrt((1-tpr)**2 + fpr**2)
    optimal_idx_closest = np.argmin(distances)
    optimal_threshold_closest = roc_thresholds[optimal_idx_closest]
    
    # Method 3: F1-score maximization
    f1_scores = []
    for threshold in np.arange(0.1, 1.0, 0.01):
        predicted = (model_scores >= threshold).astype(int)
        # Convert to collision prediction (invert the logic since model predicts no_collision_prob)
        predicted_collisions = 1 - predicted
        if np.sum(predicted_collisions) > 0 and np.sum(actual_collisions) > 0:
            precision = np.sum((predicted_collisions == 1) & (actual_collisions == 1)) / np.sum(predicted_collisions == 1)
            recall = np.sum((predicted_collisions == 1) & (actual_collisions == 1)) / np.sum(actual_collisions == 1)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append((threshold, f1))
    
    if f1_scores:
        optimal_threshold_f1 = max(f1_scores, key=lambda x: x[1])[0]
    else:
        optimal_threshold_f1 = 0.5
    
    # Test different fixed thresholds
    test_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_results = {}
    
    for threshold in test_thresholds:
        predicted = (model_scores >= threshold).astype(int)
        predicted_collisions = 1 - predicted  # Convert to collision prediction
        
        # Calculate metrics
        tn = np.sum((predicted_collisions == 0) & (actual_collisions == 0))
        fp = np.sum((predicted_collisions == 1) & (actual_collisions == 0))
        fn = np.sum((predicted_collisions == 0) & (actual_collisions == 1))
        tp = np.sum((predicted_collisions == 1) & (actual_collisions == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        threshold_results[threshold] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    
    # Save results
    results = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'optimal_thresholds': {
            'youden': float(optimal_threshold_youden),
            'closest_to_01': float(optimal_threshold_closest),
            'f1_maximization': float(optimal_threshold_f1)
        },
        'fixed_threshold_results': threshold_results,
        'data_summary': {
            'total_samples': len(model_scores),
            'actual_collisions': int(np.sum(actual_collisions)),
            'collision_rate': float(np.mean(actual_collisions)),
            'model_score_range': [float(model_scores.min()), float(model_scores.max())]
        }
    }
    
    with open(os.path.join(output_dir, "collision_prediction_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualizations (only bottom two plots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Model Score Distribution
    axes[0].hist(model_scores[actual_collisions == 0], bins=50, alpha=0.7, 
                label='No Collision', color='green')
    axes[0].hist(model_scores[actual_collisions == 1], bins=50, alpha=0.7, 
                label='Collision', color='red')
    axes[0].set_xlabel('Model Score (No Collision Probability)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Model Score Distribution')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Performance vs Threshold
    thresholds = list(threshold_results.keys())
    accuracies = [threshold_results[t]['accuracy'] for t in thresholds]
    f1_scores = [threshold_results[t]['f1'] for t in thresholds]
    
    ax2 = axes[1].twinx()
    line1 = axes[1].plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
    line2 = ax2.plot(thresholds, f1_scores, 'r-', label='F1-Score', linewidth=2)
    
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Accuracy', color='b')
    ax2.set_ylabel('F1-Score', color='r')
    axes[1].set_title('Performance vs Threshold')
    axes[1].grid(True)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1].legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "collision_prediction_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"    → Fixed threshold results:")
    for threshold, metrics in threshold_results.items():
        print(f"      - Threshold {threshold}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    
    # Save detailed confusion matrices for each threshold
    confusion_matrices = {}
    for threshold in test_thresholds:
        predicted = (model_scores >= threshold).astype(int)
        predicted_collisions = 1 - predicted
        cm = confusion_matrix(actual_collisions, predicted_collisions)
        confusion_matrices[f'threshold_{threshold}'] = cm.tolist()
    
    with open(os.path.join(output_dir, "confusion_matrices.json"), "w") as f:
        json.dump(confusion_matrices, f, indent=2)
    
    print(f"    → Saved collision_prediction_analysis.json, confusion_matrices.json, and collision_prediction_analysis.png")

def data_analysis(data_path, output_dir):
    ensure_dir(output_dir)
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples.")

    analyze_class_distribution(data, output_dir)
    analyze_pose_distributions(data, output_dir)
    analyze_grasp_pose_variance(data, output_dir)
    analyze_orientation_change_vs_success(data, output_dir)
    # analyze_surface_transitions(data, output_dir)
    visualize_random_samples(data, output_dir)
    write_report(output_dir)
    # analyze_difficulty_simple(data, output_dir)
    # analyze_model_collision_prediction(data, output_dir)

if __name__ == "__main__":
    folder = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/combined_data"
    data_path = os.path.join(folder, "train.json")
    output_path = os.path.join(folder, "analysis/train_data")
    data_analysis(data_path, output_path)

    

    # # Load the data first
    # print(f"Loading data from {data_path}...")
    # with open(data_path) as f:
    #     data = json.load(f)
    # print(f"Loaded {len(data)} samples.")

    # experiment_results_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/combined_data/combined_experiment_data.json"
    # with open(experiment_results_path) as f:
    #     experiment_results = json.load(f)
    # print(f"Loaded {len(experiment_results)} experiment results.")
    # output_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v2/analysis"
    # analyze_model_collision_prediction(experiment_results, output_path)

    

    # Save the combined data
    