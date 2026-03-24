# model_final_corners_aux.py
# FinalCornersAuxModel for (corners_24[z], aux_12) -> collision logit

import torch
import torch.nn as nn

class FinalCornersAuxModel(nn.Module):
    """
    Inputs:
      corners_24 : [B, 24]  (z-scored world corners at final pose)
      aux_k      : [B, K]   (K=12 → [t_loc_z(3), R_loc6(6 raw), dims_z(3)])
    Output:
      logits     : [B, 1]   (BCEWithLogitsLoss against collision label y∈{0,1})
    """
    def __init__(self,
                 aux_in=12,
                 corners_hidden=(128, 64),
                 aux_hidden=(64, 32),
                 head_hidden=128,
                 dropout_p=0.05,
                 use_film: bool = True,
                 two_head: bool = False):
        super().__init__()
        self.use_film = use_film
        self.two_head = two_head
        self.core_aux_dims = aux_in  # use all provided aux features (e.g., 18 incl. rf6)
        self.arm_alpha = 0.0                  # only used if two_head=True and aux_in>12

        # Corners branch
        self.corners_net = nn.Sequential(
            nn.Linear(24, corners_hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(corners_hidden[0], corners_hidden[1]),
            nn.GELU(),
        )
        self.c_ln = nn.LayerNorm(corners_hidden[1])

        # Aux-core branch
        self.aux_core_net = nn.Sequential(
            nn.Linear(self.core_aux_dims, aux_hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(aux_hidden[0], aux_hidden[1]),
            nn.GELU(),
        )
        self.a_ln = nn.LayerNorm(aux_hidden[1])

        # FiLM (aux_core → corners)
        if self.use_film:
            self.film = nn.Linear(aux_hidden[1], 2 * corners_hidden[1])

        # Fusion head (object/contact)
        self.obj_head = nn.Sequential(
            nn.Linear(corners_hidden[1] + aux_hidden[1], head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_hidden, 1),
        )

        # Optional second head for extras beyond 12 (not used here)
        arm_in = max(0, aux_in - self.core_aux_dims)
        if self.two_head and arm_in > 0:
            self.arm_head = nn.Sequential(
                nn.Linear(arm_in, 16),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(16, 1),
            )
        else:
            self.arm_head = None

    @torch.no_grad()
    def set_arm_alpha(self, alpha: float):
        self.arm_alpha = float(max(0.0, min(1.0, alpha)))

    def forward(self, corners_24: torch.Tensor, aux_k: torch.Tensor) -> torch.Tensor:
        assert corners_24.dim()==2 and corners_24.size(-1)==24, f"{corners_24.shape=}"
        assert aux_k.dim()==2, f"{aux_k.shape=}"
        B, K = aux_k.shape
        assert K >= self.core_aux_dims

        aux_core = aux_k[:, :self.core_aux_dims]
        arm_extras = aux_k[:, self.core_aux_dims:] if (self.two_head and K > self.core_aux_dims) else None

        c_feat = self.corners_net(corners_24)
        c_feat = self.c_ln(c_feat)

        a_feat = self.aux_core_net(aux_core)
        a_feat = self.a_ln(a_feat)

        if self.use_film:
            gb = self.film(a_feat)             # [B, 2*C]
            gamma, beta = torch.chunk(gb, 2, dim=-1)
            gamma = torch.tanh(gamma)          # stabilize
            c_feat = c_feat * (1.0 + gamma) + beta

        obj_logit = self.obj_head(torch.cat([c_feat, a_feat], dim=-1))  # [B,1]

        if self.two_head and (self.arm_head is not None) and (arm_extras is not None) and arm_extras.shape[1] > 0:
            arm_logit = self.arm_head(arm_extras)
            logits = obj_logit + self.arm_alpha * arm_logit
        else:
            logits = obj_logit
        return logits
    


if __name__ == "__main__":
    from dataset import FinalCornersHandDataset
    from model import FinalCornersAuxModel
    import json, torch
    mem_dir = "/home/chris/Chris/placement_ws/src/data/box_simulation/v6/data_collection/memmaps"
    stats = json.load(open(f"{mem_dir}/stats.json","r"))
    train_ds = FinalCornersHandDataset(mem_dir, normalization_stats=stats, is_training=True)
    val_ds   = FinalCornersHandDataset(mem_dir, normalization_stats=stats, is_training=False)

    model = FinalCornersAuxModel(aux_in=12, use_film=True, two_head=False)
    print(model)
# loss: BCEWithLogitsLoss (positive = collision)