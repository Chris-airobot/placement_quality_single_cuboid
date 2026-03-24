import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspPlaceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointnet = PointNetEncoder(global_feat_dim=256)  # e.g., output 256-d feature
        self.pose_mlp = nn.Sequential(
            nn.Linear(21, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(256+128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        # Output layers
        self.out_score = nn.Linear(64, 1)    # continuous quality output
        self.out_success = nn.Linear(64, 1)  # success logit output (for binary classification)

        
    def forward(self, points, pose_vec):
        obj_feat = self.pointnet(points)           # [batch, 256]
        pose_feat = self.pose_mlp(pose_vec)        # [batch, 128]
        fused = torch.cat([obj_feat, pose_feat], dim=1)  # [batch, 384]
        x = self.fusion_fc(fused)                  # [batch, 64]
        score = self.out_score(x)                  # [batch, 1] (raw score)
        success_logit = self.out_success(x)        # [batch, 1] (logit)
        success_prob = torch.sigmoid(success_logit)# [batch, 1] (0-1 probability)
        return score, success_prob
