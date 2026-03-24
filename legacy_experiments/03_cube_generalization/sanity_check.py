import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import time
from dataset import PlacementDataset
from model import create_combined_model, PointNetEncoder
from model_training import (
    get_unique_dimensions_from_dataset, 
    precompute_static_embeddings,
    StaticEmbeddingDataset,
    transform_points,
    pose_to_transformation_matrix,
    generate_box_pointcloud
)

def test_pose_transformation():
    """Test pose transformation functions"""
    print("=== Testing Pose Transformations ===")
    
    # Create test box corners (local frame)
    dx, dy, dz = 0.1, 0.2, 0.15
    corners_local = np.array([
        [-dx/2, -dy/2, -dz/2],  # corner 0
        [ dx/2, -dy/2, -dz/2],  # corner 1
        [ dx/2,  dy/2, -dz/2],  # corner 2
        [-dx/2,  dy/2, -dz/2],  # corner 3
        [-dx/2, -dy/2,  dz/2],  # corner 4
        [ dx/2, -dy/2,  dz/2],  # corner 5
        [ dx/2,  dy/2,  dz/2],  # corner 6
        [-dx/2,  dy/2,  dz/2],  # corner 7
    ])
    
    # Test pose: translate by [1, 2, 3] and rotate 45 degrees around Z
    test_pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.3827, 0.9239])  # 45 deg rotation
    
    # Transform corners
    corners_world = transform_points(torch.tensor(corners_local), test_pose)
    
    print(f"Original box dimensions: {dx:.3f} x {dy:.3f} x {dz:.3f}")
    print(f"Test pose: pos=[{test_pose[0]:.2f}, {test_pose[1]:.2f}, {test_pose[2]:.2f}]")
    print(f"Local frame center: {np.mean(corners_local, axis=0)}")
    print(f"World frame center: {np.mean(corners_world.numpy(), axis=0)}")
    
    # Check if dimensions are preserved
    local_dims = np.max(corners_local, axis=0) - np.min(corners_local, axis=0)
    world_dims = np.max(corners_world.numpy(), axis=0) - np.min(corners_world.numpy(), axis=0)
    print(f"Local dims: [{local_dims[0]:.6f}, {local_dims[1]:.6f}, {local_dims[2]:.6f}]")
    print(f"World dims: [{world_dims[0]:.6f}, {world_dims[1]:.6f}, {world_dims[2]:.6f}]")
    
    # Check if transformation is reasonable (center moved correctly)
    expected_center = test_pose[:3]  # [1, 2, 3]
    actual_center = np.mean(corners_world.numpy(), axis=0)
    center_correct = np.allclose(expected_center, actual_center, atol=1e-6)
    print(f"✓ Center transformation correct: {center_correct}")
    return center_correct

def test_dataset_loading(data_path):
    """Test dataset loading and basic properties"""
    print("\n=== Testing Dataset Loading ===")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path not found: {data_path}")
        return False
    
    # Test base dataset
    dataset = PlacementDataset(data_path)
    print(f"✓ Loaded dataset with {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    corners, grasp, init, final, label = sample
    print(f"Sample 0 shapes:")
    print(f"  - Corners: {corners.shape}")
    print(f"  - Grasp: {grasp.shape}")
    print(f"  - Init pose: {init.shape}")
    print(f"  - Final pose: {final.shape}")
    print(f"  - Label: {label.shape}")
    
    # Check label range
    success_labels = [dataset[i][4][0].item() for i in range(min(100, len(dataset)))]
    collision_labels = [dataset[i][4][1].item() for i in range(min(100, len(dataset)))]
    print(f"Success label range: [{min(success_labels):.1f}, {max(success_labels):.1f}]")
    print(f"Collision label range: [{min(collision_labels):.1f}, {max(collision_labels):.1f}]")
    
    return True

def test_static_embeddings(data_path, device):
    """Test static embedding computation"""
    print("\n=== Testing Static Embeddings ===")
    
    # Get unique dimensions
    unique_dims = get_unique_dimensions_from_dataset(data_path)
    print(f"Found {len(unique_dims)} unique dimensions")
    print(f"First 5 dimensions: {unique_dims[:5]}")
    
    # Test with subset for speed
    test_dims = unique_dims[:3]
    print(f"Testing with {len(test_dims)} dimensions")
    
    # Pre-compute embeddings
    start_time = time.time()
    embeddings_dict = precompute_static_embeddings(test_dims, device, cache_path=None)
    elapsed = time.time() - start_time
    
    print(f"✓ Computed embeddings in {elapsed:.2f}s")
    
    # Check embedding properties
    for dims, embedding in embeddings_dict.items():
        print(f"  Dims {dims}: embedding shape {embedding.shape}")
        
        # Check for reasonable values
        embed_mean = embedding.mean().item()
        embed_std = embedding.std().item()
        print(f"    Mean: {embed_mean:.4f}, Std: {embed_std:.4f}")
        
        # Embeddings should not be all zeros or identical
        assert not torch.allclose(embedding, torch.zeros_like(embedding)), "Embedding is all zeros!"
        
    return embeddings_dict

def test_combined_dataset(data_path, embeddings_dict, device):
    """Test the combined dataset with static embeddings"""
    print("\n=== Testing Combined Dataset ===")
    
    base_dataset = PlacementDataset(data_path)
    combined_dataset = StaticEmbeddingDataset(base_dataset, embeddings_dict, device)
    
    print(f"✓ Created combined dataset with {len(combined_dataset)} samples")
    
    # Test sample
    sample = combined_dataset[0]
    corners_world, grasp, init, final, label, static_embedding = sample
    
    print(f"Combined sample shapes:")
    print(f"  - Corners (world): {corners_world.shape}")
    print(f"  - Static embedding: {static_embedding.shape}")
    
    # Compare local vs world corners
    base_sample = base_dataset[0]
    corners_local = base_sample[0]
    final_pose = base_sample[3]
    
    print(f"Local corners center: {torch.mean(corners_local, dim=0)}")
    print(f"World corners center: {torch.mean(corners_world, dim=0)}")
    
    # Manually transform and compare
    corners_manual = transform_points(corners_local, final_pose)
    diff = torch.abs(corners_world - corners_manual).max()
    print(f"✓ Transformation consistency: max diff = {diff:.8f}")
    
    return combined_dataset

def test_model_architecture(device):
    """Test model architecture and forward pass"""
    print("\n=== Testing Model Architecture ===")
    
    model = create_combined_model().to(device)
    print(f"✓ Created combined model")
    
    # Test input shapes
    batch_size = 4
    points = torch.randn(batch_size, 1024, 3).to(device)
    corners = torch.randn(batch_size, 8, 3).to(device)
    grasp_pose = torch.randn(batch_size, 7).to(device)
    init_pose = torch.randn(batch_size, 7).to(device)
    final_pose = torch.randn(batch_size, 7).to(device)
    
    print(f"Test input shapes:")
    print(f"  - Points: {points.shape}")
    print(f"  - Corners: {corners.shape}")
    print(f"  - Poses: {grasp_pose.shape} each")
    
    # Forward pass
    with torch.no_grad():
        logit_c = model(
            points=points, 
            corners=corners,
            grasp_pose=grasp_pose,
            init_pose=init_pose, 
            final_pose=final_pose
        )
    
    print(f"✓ Forward pass successful")
    print(f"Output shapes: logit_c={logit_c.shape}")
    
    # Check output ranges
    print(f"Collision logits range: [{logit_c.min():.3f}, {logit_c.max():.3f}]")
    
    # Test individual components
    print("\nTesting individual encoders:")
    corner_feat = model.corner_encoder(corners)
    print(f"  Corner features: {corner_feat.shape}")
    
    if hasattr(model, 'pointnet_encoder'):
        pointnet_feat = model.pointnet_encoder(points)
        print(f"  PointNet features: {pointnet_feat.shape}")
    
    return model

def test_memory_usage(combined_dataset, model, device):
    """Test memory usage during training"""
    print("\n=== Testing Memory Usage ===")
    
    from torch.utils.data import DataLoader
    
    # Small batch test
    loader = DataLoader(combined_dataset, batch_size=8, shuffle=False)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    model.train()
    for i, batch in enumerate(loader):
        if i >= 2:  # Test just a few batches
            break
            
        corners_world, grasp, init, final, label, static_embedding = batch
        corners_world = corners_world.to(device)
        grasp = grasp.to(device)
        init = init.to(device)
        final = final.to(device)
        static_embedding = static_embedding.to(device)
        
        # Manual forward pass (like in training)
        corner_feat = model.corner_encoder(corners_world)
        batch_size = corners_world.size(0)
        # Ensure static_embedding is [batch_size, 256]
        if static_embedding.dim() == 3:
            static_embedding = static_embedding.squeeze(1)  # [B, 1, 256] -> [B, 256]
        pointnet_feat = static_embedding.expand(batch_size, -1)
        obj_feat = torch.cat([pointnet_feat, corner_feat], dim=1)
        pose_input = torch.cat([grasp, init, final], dim=1)
        pose_feat = model.pose_encoder(pose_input)
        fused = torch.cat([obj_feat, pose_feat], dim=1)
        hidden = model.fusion_fc(fused)
        logit_s = model.out_success(hidden)
        logit_c = model.out_collision(hidden)
        
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"Batch {i}: GPU memory: {current_memory:.1f} MB")
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"✓ Peak GPU memory: {peak_memory:.1f} MB")

def visualize_transformation_sample(combined_dataset):
    """Visualize a sample transformation"""
    print("\n=== Visualizing Sample Transformation ===")
    
    # Get base dataset for comparison
    base_dataset = combined_dataset.base_dataset
    
    # Get sample
    idx = 0
    corners_local = base_dataset[idx][0]
    final_pose = base_dataset[idx][3]
    corners_world, _, _, _, _, _ = combined_dataset[idx]
    
    # Create plot
    fig = plt.figure(figsize=(12, 5))
    
    # Local frame
    ax1 = fig.add_subplot(121, projection='3d')
    corners_np = corners_local.numpy()
    ax1.scatter(corners_np[:, 0], corners_np[:, 1], corners_np[:, 2], c='blue', s=50)
    ax1.set_title('Local Frame')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # World frame
    ax2 = fig.add_subplot(122, projection='3d')
    corners_world_np = corners_world.numpy()
    ax2.scatter(corners_world_np[:, 0], corners_world_np[:, 1], corners_world_np[:, 2], c='red', s=50)
    ax2.set_title('World Frame (Final Pose)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('transformation_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved transformation visualization to 'transformation_visualization.png'")
    plt.close()

def run_mini_training_test(combined_dataset, model, device):
    """Run a mini training test"""
    print("\n=== Mini Training Test ===")
    
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    
    # Small dataloader
    loader = DataLoader(combined_dataset, batch_size=4, shuffle=True)
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i, batch in enumerate(loader):
        if i >= 3:  # Just test a few batches
            break
            
        corners_world, grasp, init, final, label, static_embedding = batch
        corners_world = corners_world.to(device)
        grasp = grasp.to(device)
        init = init.to(device)
        final = final.to(device)
        label = label.to(device)
        static_embedding = static_embedding.to(device)
        
        sl = label[:, 0].unsqueeze(1)
        cl = label[:, 1].unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Manual forward pass
        corner_feat = model.corner_encoder(corners_world)
        batch_size = corners_world.size(0)
        # Ensure static_embedding is [batch_size, 256]
        if static_embedding.dim() == 3:
            static_embedding = static_embedding.squeeze(1)  # [B, 1, 256] -> [B, 256]
        pointnet_feat = static_embedding.expand(batch_size, -1)
        obj_feat = torch.cat([pointnet_feat, corner_feat], dim=1)
        pose_input = torch.cat([grasp, init, final], dim=1)
        pose_feat = model.pose_encoder(pose_input)
        fused = torch.cat([obj_feat, pose_feat], dim=1)
        hidden = model.fusion_fc(fused)
        logit_s = model.out_success(hidden)
        logit_c = model.out_collision(hidden)
        
        loss_s = criterion(logit_s, sl)
        loss_c = criterion(logit_c, cl)
        loss = 0.5 * loss_s + 0.5 * loss_c
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        print(f"  Batch {i}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"✓ Mini training test completed. Avg loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    """Run comprehensive sanity check"""
    print("🔍 COMPREHENSIVE SANITY CHECK")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration - using test.json for faster sanity check
    data_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v3/data_collection/combined_data/test.json"
    
    success_count = 0
    total_tests = 8
    
    try:
        # Test 1: Pose transformations
        if test_pose_transformation():
            success_count += 1
        
        # Test 2: Dataset loading
        if test_dataset_loading(data_path):
            success_count += 1
        
        # Test 3: Static embeddings
        embeddings_dict = test_static_embeddings(data_path, device)
        if embeddings_dict:
            success_count += 1
        
        # Test 4: Combined dataset
        combined_dataset = test_combined_dataset(data_path, embeddings_dict, device)
        if combined_dataset:
            success_count += 1
        
        # Test 5: Model architecture
        model = test_model_architecture(device)
        if model:
            success_count += 1
        
        # Test 6: Memory usage
        test_memory_usage(combined_dataset, model, device)
        success_count += 1
        
        # Test 7: Visualization
        visualize_transformation_sample(combined_dataset)
        success_count += 1
        
        # Test 8: Mini training
        if run_mini_training_test(combined_dataset, model, device) < 10.0:  # Reasonable loss
            success_count += 1
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 SANITY CHECK SUMMARY")
    print(f"Passed: {success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED! Your setup looks good to go!")
    elif success_count >= total_tests * 0.8:
        print("⚠️  Most tests passed. Minor issues to address.")
    else:
        print("❌ Several tests failed. Please review the errors above.")
    
    return success_count == total_tests

if __name__ == "__main__":
    main() 