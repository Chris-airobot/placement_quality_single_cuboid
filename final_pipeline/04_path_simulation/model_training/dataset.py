# dataset.py
import os
import json
import torch
from torch.utils.data import Dataset

class KinematicFeasibilityDataset(Dataset):
    """
    Dataset reading a single JSON file containing a list of samples.
    Each sample is a dict with keys:
      - 'grasp_pose': [x,y,z,qx,qy,qz,qw]
      - 'initial_object_pose': [z,qx,qy,qz,qw]
      - 'final_object_pose': [x,y,z,qx,qy,qz,qw]
      - 'success_label': float {0.0,1.0}
      - 'collision_label': float {0.0,1.0}
      - 'surfaces': "i_j"
    """
    def __init__(self, data_path: str):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        N = len(self.data)
        # Pre-allocate tensors
        self.grasp       = torch.empty((N, 7), dtype=torch.float32)
        self.init_raw    = torch.empty((N, 5), dtype=torch.float32)  # [z, qx,qy,qz,qw]
        self.final_raw   = torch.empty((N, 5), dtype=torch.float32)
        self.success     = torch.empty((N,),   dtype=torch.float32)
        self.collision   = torch.empty((N,),   dtype=torch.float32)
        self.const_xy   = torch.tensor([0.35, 0.0], dtype=torch.float32)

        for i, s in enumerate(self.data):
            self.grasp[i]     = torch.tensor(s['grasp_pose'],            dtype=torch.float32)
            self.init_raw[i]  = torch.tensor(s['initial_object_pose'],   dtype=torch.float32)
            self.final_raw[i] = torch.tensor(s['final_object_pose'],     dtype=torch.float32)
            self.success[i]   = float(s['success_label'])
            self.collision[i] = float(s['collision_label'])

        # self.object_pcd = object_pcd            # already a Tensor [num_pts, 3]

    def __len__(self):
        return self.grasp.size(0)

    def __getitem__(self, idx):
        # points — same for every sample
        grasp  = self.grasp[idx]
        init   = torch.cat([self.const_xy, self.init_raw[idx]])
        final  = torch.cat([self.const_xy, self.final_raw[idx]])
        sl     = self.success[idx]
        cl     = self.collision[idx]
        return grasp, init, final, sl, cl