# # pointnet2_models.py
# import torch
# import torch.nn as nn



# if __name__ == "__main__":
#     encoder = PointNetEncoder(global_feat_dim=256)
#     points = torch.rand(8, 1024, 3)  # batch_size=8, N_points=1024
#     global_features = encoder(points)

#     print(global_features.shape)  # should output: torch.Size([8, 256])
