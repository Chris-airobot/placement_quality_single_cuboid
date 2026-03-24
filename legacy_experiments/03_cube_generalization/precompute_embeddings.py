import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from dataset import generate_box_pointcloud, transform_points_to_world
import psutil

# Add PointNet++ path
pointnet_path = '/home/chris/Chris/placement_ws/src/Pointnet_Pointnet2_pytorch'
sys.path.append(pointnet_path)
sys.path.append(os.path.join(pointnet_path, 'models'))

# Import after adding paths
from pointnet2.pointnet2_cls_ssg import get_model
from collections import OrderedDict

def norm_quat(qw,qx,qy,qz):
    # normalize length
    n = float((qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5)
    if n > 0.0:
        qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    # force a canonical sign (w >= 0) to avoid ±q duplicates
    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return (qw, qx, qy, qz)

def precompute_embeddings(
    data_path,
    output_file,
    batch_size: int = 384,
    chunk_size: int = 5000,
    combine: bool = False,
    pre_count: bool = False,
    num_points: int = 512,
    use_amp: bool = True,
):
    """Precompute world-frame embeddings with streaming to avoid high RAM/CPU.

    - Streams JSON samples from data_path (array of objects) using ijson if available.
    - Processes in batches on GPU and writes shard .npy files per chunk.
    - If combine=True, consolidates shards into a single memmap .npy without loading all in RAM.
    """
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Streaming reader setup
    try:
        import ijson  # type: ignore
        use_ijson = True
    except Exception:
        use_ijson = False

    # Optional pre-count for ETA if requested
    total_items = None
    if pre_count:
        try:
            if use_ijson:
                c = 0
                with open(data_path, "rb") as f:
                    for _ in ijson.items(f, "item"):
                        c += 1
                total_items = c
            else:
                # fallback rough count (loads file; skip if too big)
                with open(data_path, "r") as f:
                    total_items = len(json.load(f))
        except Exception:
            total_items = None
    
    # Load pre-trained PointNet++ model
    pretrained_model_path = '/home/chris/Chris/placement_ws/src/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    
    # Create PointNet++ model
    pointnet = get_model(num_class=40, normal_channel=False).to(device)  # 40 classes for ModelNet40
    torch.backends.cudnn.benchmark = True
    
    # Load pre-trained weights
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    pointnet.load_state_dict(checkpoint['model_state_dict'])
    pointnet.eval()
    
    
    print(f"Loaded pre-trained PointNet++ from {pretrained_model_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Precompute unit cube point cloud once and scale per-sample (vectorized)
    base_local_np = generate_box_pointcloud(1.0, 1.0, 1.0, num_points=num_points)
    base_local = torch.from_numpy(base_local_np).to(torch.float32)

    # Create temporary files for incremental saving
    temp_dir = os.path.dirname(output_file)
    temp_prefix = os.path.basename(output_file).replace('.npy', '_chunk_')
    
    chunk_files = []
    processed = 0
    
    def _iter_samples(path):
        if use_ijson:
            with open(path, "rb") as f:
                for obj in ijson.items(f, "item"):
                    yield obj
        else:
            # Fallback: not memory-friendly for huge files; warn
            with open(path, "r") as f:
                for obj in json.load(f):
                    yield obj

    with torch.no_grad():
        pbar = tqdm(total=total_items, desc="Embedding", unit="samples")
        chunk_samples = []  # list of (dx,dy,dz, init_pose)

        def make_key(dx,dy,dz, init_p):
            px,py,pz, qw,qx,qy,qz = [float(v) for v in init_p]
            qw,qx,qy,qz = norm_quat(qw,qx,qy,qz)
            return (
                round(dx,5), round(dy,5), round(dz,5),
                round(px,5), round(py,5), round(pz,5),
                round(qw,7), round(qx,7), round(qy,7), round(qz,7),
            )

        def process_chunk(samples, chunk_idx):
            if not samples:
                return
            # Dedup by world pose key
            key_to_uindex = {}
            unique_dims = []
            unique_poses = []
            map_idx = []
            for (dx,dy,dz, init_p) in samples:
                key = make_key(dx,dy,dz, init_p)
                u = key_to_uindex.get(key)
                if u is None:
                    u = len(unique_dims)
                    key_to_uindex[key] = u
                    unique_dims.append((dx,dy,dz))
                    unique_poses.append(init_p)
                map_idx.append(u)

            # Compute embeddings for uniques in big batches
            uniq = len(unique_dims)
            all_parts = []
            for start in range(0, uniq, batch_size):
                end = min(start + batch_size, uniq)
                dims = torch.tensor(unique_dims[start:end], device=device, dtype=torch.float32)
                poses = torch.tensor(unique_poses[start:end], device=device, dtype=torch.float32)
                pos = poses[:, :3]
                q = poses[:, 3:7]
                w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
                xx, yy, zz = x*x, y*y, z*z
                xy, xz, yz = x*y, x*z, y*z
                wx, wy, wz = w*x, w*y, w*z
                R00 = 1 - 2*(yy + zz); R01 = 2*(xy - wz); R02 = 2*(xz + wy)
                R10 = 2*(xy + wz); R11 = 1 - 2*(xx + zz); R12 = 2*(yz - wx)
                R20 = 2*(xz - wy); R21 = 2*(yz + wx); R22 = 1 - 2*(xx + yy)
                R = torch.stack([
                    torch.stack([R00, R01, R02], dim=-1),
                    torch.stack([R10, R11, R12], dim=-1),
                    torch.stack([R20, R21, R22], dim=-1),
                ], dim=-2)
                local = base_local.to(device).unsqueeze(0) * dims.unsqueeze(1)
                world = torch.einsum('bij,bnj->bni', R, local) + pos.unsqueeze(1)
                batch_pcds = world.transpose(1, 2)
                batch_pcds = batch_pcds.to(torch.float32)
                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        out = pointnet(batch_pcds)
                else:
                    out = pointnet(batch_pcds)
                emb = out[1].squeeze(-1).cpu().numpy().astype(np.float16)
                all_parts.append(emb)
            uniques_array = np.concatenate(all_parts, axis=0)

            # Map back to full chunk order
            d = uniques_array.shape[1]
            chunk_array = np.empty((len(samples), d), dtype=np.float16)
            for i, u in enumerate(map_idx):
                chunk_array[i] = uniques_array[u]

            chunk_file = os.path.join(temp_dir, f"{temp_prefix}{chunk_idx:04d}.npy")
            np.save(chunk_file, chunk_array)
            chunk_files.append(chunk_file)
            print(f"Saved chunk {chunk_idx+1}: {chunk_file} (uniques={uniq}/{len(samples)})")

        chunk_idx = 0
        for sample in _iter_samples(data_path):
            dims = sample.get("object_dimensions")
            init_pose = sample.get("initial_object_pose")
            if dims is None or init_pose is None:
                continue
            chunk_samples.append((float(dims[0]), float(dims[1]), float(dims[2]), np.array(init_pose, dtype=np.float64)))
            processed += 1
            if pbar is not None:
                pbar.update(1)
            if len(chunk_samples) >= chunk_size:
                process_chunk(chunk_samples, chunk_idx)
                chunk_idx += 1
                chunk_samples.clear()
        if chunk_samples:
            process_chunk(chunk_samples, chunk_idx)
        if pbar is not None:
            pbar.close()
    
    if combine:
        # Combine shards into a single memmap without loading all in RAM
        print("Combining shards with memmap...")
        # Determine total rows and feature dim
        n_total = 0
        feat_dim = None
        for cf in chunk_files:
            a = np.load(cf, mmap_mode='r')
            n_total += a.shape[0]
            feat_dim = a.shape[1] if feat_dim is None else feat_dim
        mm = np.memmap(output_file, dtype=np.float16, mode='w+', shape=(n_total, feat_dim))
        pos = 0
        for cf in chunk_files:
            a = np.load(cf, mmap_mode='r')
            mm[pos:pos+a.shape[0]] = a
            pos += a.shape[0]
        del mm
        print("Cleaning up temporary files...")
        for cf in chunk_files:
            try:
                os.remove(cf)
            except Exception:
                pass
        file_size_gb = os.path.getsize(output_file) / (1024**3)
        print(f"Saved {processed} embeddings to {output_file} (~{file_size_gb:.2f} GB)")
    else:
        # Keep shards; write an index file listing them
        index_path = os.path.splitext(output_file)[0] + "_shards.txt"
        with open(index_path, "w") as f:
            for cf in chunk_files:
                f.write(cf + "\n")
        print(f"Saved {processed} embeddings into {len(chunk_files)} shards. Index: {index_path}")

def precompute_experiment_embeddings(experiment_file, output_file):
    """Precompute embeddings for experiment generation data"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load experiment data
    with open(experiment_file, 'r') as f:
        experiments = json.load(f)
    print(f"Loaded {len(experiments)} experiments from {experiment_file}")
    
    # Load pre-trained PointNet++ model
    pretrained_model_path = '/home/chris/Chris/placement_ws/src/Pointnet_Pointnet2_pytorch/log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    
    # Create PointNet++ model
    pointnet = get_model(num_class=40, normal_channel=False).to(device)
    
    # Load pre-trained weights
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    pointnet.load_state_dict(checkpoint['model_state_dict'])
    pointnet.eval()
    
    print(f"Loaded pre-trained PointNet++ from {pretrained_model_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Batch processing setup
    batch_size = 384
    chunk_size = 5000
    
    # Create temporary files for incremental saving
    temp_dir = os.path.dirname(output_file)
    temp_prefix = os.path.basename(output_file).replace('.npy', '_chunk_')
    
    chunk_files = []
    
    with torch.no_grad():
        for chunk_idx, chunk_start in enumerate(tqdm(range(0, len(experiments), chunk_size), desc="Processing chunks")):
            chunk_end = min(chunk_start + chunk_size, len(experiments))
            
            # Small chunk array
            chunk_embeddings = []
            
            for batch_start in range(chunk_start, chunk_end, batch_size):
                batch_end = min(batch_start + batch_size, chunk_end)
                current_batch_size = batch_end - batch_start
                
                # Prepare batch tensors
                batch_pcds = torch.zeros(current_batch_size, 1024, 3, device=device)
                
                # Process batch
                for i, idx in enumerate(range(batch_start, batch_end)):
                    exp_data = experiments[idx]
                    
                    # Extract object dimensions and initial pose from experiment format
                    dx, dy, dz = exp_data['object_dimensions']
                    initial_pose = exp_data['initial_object_pose']  # [x, y, z, qw, qx, qy, qz]

                    qw,qx,qy,qz = [float(v) for v in initial_pose[3:7]]
                    qw,qx,qy,qz = norm_quat(qw,qx,qy,qz)
                    initial_pose = [float(exp_data['initial_object_pose'][0]),
                                    float(exp_data['initial_object_pose'][1]),
                                    float(exp_data['initial_object_pose'][2]),
                                    qw,qx,qy,qz]
                    
                    # Generate local point cloud for this object
                    local_pcd = generate_box_pointcloud(dx, dy, dz, num_points=1024)
                    
                    # Transform to world frame using initial pose
                    world_pcd = transform_points_to_world(local_pcd, np.array(initial_pose))
                    
                    # Add to batch (PointNet++ expects [B, N, 3] format)
                    batch_pcds[i] = torch.from_numpy(world_pcd).float()
                
                # PointNet++ expects input in [B, 3, N] format, so transpose
                batch_pcds = batch_pcds.transpose(1, 2)  # [B, N, 3] -> [B, 3, N]
                
                # Process entire batch at once
                batch_output = pointnet(batch_pcds)  # Returns (logits, features)
                batch_embeddings = batch_output[1].squeeze(-1)  # Extract 1024-D features
                chunk_embeddings.append(batch_embeddings.cpu().numpy())
            
            # Save this chunk immediately
            chunk_array = np.concatenate(chunk_embeddings, axis=0).astype(np.float16)
            chunk_file = os.path.join(temp_dir, f"{temp_prefix}{chunk_idx:04d}.npy")
            np.save(chunk_file, chunk_array)
            chunk_files.append(chunk_file)
            
            print(f"Saved chunk {chunk_idx+1}/{(len(experiments) + chunk_size - 1) // chunk_size}: {chunk_file}")
    
    # Combine all chunk files into final file
    print("Combining all chunks into final file...")
    all_embeddings = []
    for chunk_file in chunk_files:
        chunk_data = np.load(chunk_file)
        all_embeddings.append(chunk_data)
    
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_file, final_embeddings)
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"✅ Saved {len(experiments)} embeddings to {output_file}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Embeddings shape: {final_embeddings.shape}")
    
    return output_file

def test_small_subset():
    """Test with just 100 samples to validate everything works"""
    print("=== TESTING WITH 100 SAMPLES ===")
    
    # Load small subset
    with open("/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/combined_data/val.json", 'r') as f:
        full_data = json.load(f)
    
    small_data = full_data[:100]  # Just 100 samples
    test_path = "test_100.json"
    with open(test_path, 'w') as f:
        json.dump(small_data, f)
    
    # Copy processed_data directory for test
    import shutil
    if not os.path.exists("processed_data"):
        shutil.copytree("/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/processed_data", "processed_data")
    
    # Test preprocessing
    test_output = "./test_embeddings.npy"
    precompute_embeddings(test_path, test_output)
    
    # Check file size
    file_size_mb = os.path.getsize(test_output) / (1024**2)
    expected_size_mb = 100 * 1024 * 2 / (1024**2)  # float16 = 2 bytes
    print(f"File size: {file_size_mb:.2f} MB (expected: {expected_size_mb:.2f} MB)")
    
    # Test loading
    embeddings = np.load(test_output)
    print(f"Loaded embeddings shape: {embeddings.shape}")
    
    # Validate file size
    file_size_mb = os.path.getsize(test_output) / (1024**2)
    expected_size_mb = 100 * 1024 * 2 / (1024**2)  # float16 = 2 bytes
    print(f"File size: {file_size_mb:.2f} MB (expected: {expected_size_mb:.2f} MB)")
    
    # if abs(file_size_mb - expected_size_mb) > 0.1:
    #     print(f"❌ File size wrong! Expected ~{expected_size_mb:.2f}MB, got {file_size_mb:.2f}MB")
    #     return False
    
    # Validate embedding content
    print("Validating embedding content...")
    if embeddings.shape != (100, 1024):
        print(f"❌ Wrong shape! Expected (100, 1024), got {embeddings.shape}")
        return False
    
    if not np.isfinite(embeddings).all():
        print("❌ Embeddings contain NaN or inf values!")
        return False
    
    # Check embedding values are reasonable (not all zeros, reasonable range)
    if np.allclose(embeddings, 0):
        print("❌ All embeddings are zero!")
        return False
    
    embedding_std = np.std(embeddings)
    if embedding_std < 0.01:
        print(f"❌ Embeddings too uniform! std={embedding_std:.6f}")
        return False
    
    print(f"✅ Embeddings look good! std={embedding_std:.6f}")
    
    # Test dataset loading
    from dataset import WorldFrameDataset
    test_dataset = WorldFrameDataset(test_path, test_output)
    print(f"Dataset length: {len(test_dataset)}")
    
    # Test one sample
    sample = test_dataset[0]
    print(f"Sample shapes: corners={sample[0].shape}, embedding={sample[1].shape}")
    
    # Test model forward pass
    print("Testing model forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import create_combined_model
    model = create_combined_model().to(device)
    model.eval()
    
    with torch.no_grad():
        corners, embedding, grasp, init, final, label = sample
        corners, embedding, grasp, init, final = [t.to(device) for t in (corners, embedding, grasp, init, final)]
        
        logits = model(embedding.unsqueeze(0), corners.unsqueeze(0), grasp.unsqueeze(0), init.unsqueeze(0), final.unsqueeze(0))
        print(f"✅ Model forward pass works! Output shape: {logits.shape}")
    
    # Cleanup
    os.remove(test_path)
    # os.remove(test_output)  # Commented out to keep test embeddings for inspection
    shutil.rmtree("processed_data")
    print("✅ All tests passed! Ready for full dataset.")
    print(f"Test embeddings saved to: {test_output}")
    return True

def validate_full_dataset():
    """Validate the full validation dataset embeddings"""
    print("=== VALIDATING FULL VALIDATION DATASET ===")
    
    val_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/data_collection/combined_data/val.json"
    embeddings_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v4/embeddings/val_embeddings.npy"
    
    # Check file exists
    if not os.path.exists(embeddings_file):
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return False
    
    # Load embeddings
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Check file size
    file_size_gb = os.path.getsize(embeddings_file) / (1024**3)
    expected_size_gb = embeddings.shape[0] * 1024 * 4 / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB (expected: {expected_size_gb:.2f} GB)")
    
    # if abs(file_size_gb - expected_size_gb) > 0.1:
    #     print(f"❌ File size wrong!")
    #     return False
    
    # Validate content
    print("Validating embedding content...")
    if not np.isfinite(embeddings).all():
        print("❌ Embeddings contain NaN or inf values!")
        return False
    
    if np.allclose(embeddings, 0):
        print("❌ All embeddings are zero!")
        return False
    
    embedding_std = np.std(embeddings)
    print(f"Embedding statistics:")
    print(f"  - Mean: {np.mean(embeddings):.6f}")
    print(f"  - Std: {embedding_std:.6f}")
    print(f"  - Min: {np.min(embeddings):.6f}")
    print(f"  - Max: {np.max(embeddings):.6f}")
    
    if embedding_std < 0.01:
        print(f"❌ Embeddings too uniform!")
        return False
    
    # Test dataset loading
    print("Testing dataset loading...")
    from dataset import WorldFrameDataset
    dataset = WorldFrameDataset(val_path, embeddings_file)
    print(f"Dataset length: {len(dataset)}")
    
    # Test a few samples
    print("Testing sample loading...")
    for i in range(3):
        sample = dataset[i]
        corners, embedding, grasp, init, final, label = sample
        print(f"  Sample {i}: corners={corners.shape}, embedding={embedding.shape}, label={label}")
    
    # Test model forward pass
    print("Testing model forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model import create_combined_model
    model = create_combined_model().to(device)
    model.eval()
    
    with torch.no_grad():
        sample = dataset[0]
        corners, embedding, grasp, init, final, label = sample
        corners, embedding, grasp, init, final = [t.to(device) for t in (corners, embedding, grasp, init, final)]
        
        logits = model(embedding.unsqueeze(0), corners.unsqueeze(0), grasp.unsqueeze(0), init.unsqueeze(0), final.unsqueeze(0))
        print(f"✅ Model forward pass works! Output shape: {logits.shape}")
    
    print("✅ Full validation dataset is correct!")
    return True

if __name__ == "__main__":
    import sys
    test = False
    validate = False
    experiment = False  # New flag for experiment data
    
    if test:
        test_small_subset()
    elif validate:
        validate_full_dataset()
    elif experiment:
        # Process experiment generation data
        experiment_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiments.json"
        output_file = "/home/chris/Chris/placement_ws/src/placement_quality/cube_generalization/experiment_embeddings.npy"
        
        precompute_experiment_embeddings(experiment_file, output_file)
    else:
        # Original functionality
        data_path = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/data_collection/combined_data/train.json"
        output_file = "/home/chris/Chris/placement_ws/src/data/box_simulation/v5/embeddings/train/train_embeddings.npy"
        
        precompute_embeddings(
            data_path=data_path,
            output_file=output_file,
            batch_size=1536,        # try 1536 if memory allows
            chunk_size=20000,
            combine=False,
            pre_count=False,
            num_points=1024,
            use_amp=True,
        )