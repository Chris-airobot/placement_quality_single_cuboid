import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random

def compute_stability_score(pos_diff, ori_diff, shift_pos, shift_ori, contacts,
                            pos_max, ori_max, shift_pos_max, shift_ori_max, contacts_max, params=None):
    """
    using the provided upper bounds, then combining them with weighted penalties.
    """
    pos_norm       = pos_diff / pos_max
    ori_norm       = ori_diff / ori_max
    shift_pos_norm = shift_pos / shift_pos_max
    shift_ori_norm = shift_ori / shift_ori_max
    contacts_norm  = contacts / contacts_max
    
    penalty = (params['pos_weight'] * pos_norm + 
               params['pos_weight'] * ori_norm + 
               params['shift_weight'] * shift_pos_norm + 
               params['shift_weight'] * shift_ori_norm + 
               params['conatct_weight'] * contacts_norm)
    stability = -penalty  # No clamping is applied here
    return stability


# -------------------------------------------------
# ADD THIS HELPER TO MAP [-2,0] --> [-1,1]:
def to_tanh_range(value, y_min, y_max):
    """Linearly map value in [y_min,y_max] to [-1,1]."""
    return 2.0 * (value - y_min) / (y_max - y_min) - 1.0
# -------------------------------------------------

class MyStabilityDataset(Dataset):
    def __init__(self, all_entries):
        """
        all_entries is a list of dicts, each with:
          - "inputs": { ... }  
          - "outputs": { ... }
        """
        self.samples = []
        self.gripper_reference = False
        self.data_params = None

        with open("/home/chris/Chris/placement_ws/src/data/processed_data/parameters.json", 'r') as f1:
            self.data_params = json.load(f1)


        for entry in all_entries:
            # Encode inputs
            x = self.encode_inputs(entry["inputs"])
            # Encode outputs as a tuple: (classification label, regression label)
            labels = self.encode_outputs(entry["outputs"])
            if x is not None and labels is not None:
                self.samples.append((x, labels))

    def encode_inputs(self, inputs):
        """
        Flatten and concatenate input features into a 1D tensor.
        """
        init_pos = np.array(inputs["cube_initial_position"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_initial_rel_position"], dtype=np.float32)
        init_ori = np.array(inputs["cube_initial_orientation"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_initial_rel_orientation"], dtype=np.float32)
        targ_pos = np.array(inputs["cube_target_position"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_target_rel_position"], dtype=np.float32)
        targ_ori = np.array(inputs["cube_target_orientation"], dtype=np.float32) if not self.gripper_reference else np.array(inputs["cube_target_rel_orientation"], dtype=np.float32)
        grasp_pos = np.array(inputs["grasp_position"], dtype=np.float32) if inputs.get("grasp_position") is not None else init_pos
        grasp_ori = np.array(inputs["grasp_orientation"], dtype=np.float32) if inputs.get("grasp_orientation") is not None else init_ori

        x_concat = np.concatenate([
            grasp_pos, grasp_ori,
            init_pos, init_ori,
            targ_pos, targ_ori
        ])
        return torch.from_numpy(x_concat)

    def encode_outputs(self, outputs: dict):
        """
        Encode outputs into two labels:
          - Classification: feasibility (0 for failure, 1 for success)
          - Regression: stability score in [0, 1]
        """
        is_fail = outputs.get("grasp_unsuccessful", False) or outputs.get("bad", False)
        feasibility_label = 0 if is_fail else 1
        
        
        def get_valid_float(outputs: dict, key, default):
            value = outputs.get(key, default)
            return float(value) if isinstance(value, float) else float(default)

        pos_diff   = get_valid_float(outputs, "position_difference", self.data_params["position_difference"])
        ori_diff   = get_valid_float(outputs, "orientation_difference", self.data_params["orientation_difference"])
        shift_pos  = get_valid_float(outputs, "shift_position", self.data_params["shift_position"])
        shift_ori  = get_valid_float(outputs, "shift_orientation", self.data_params["shift_orientation"])
        contacts   = get_valid_float(outputs, "contacts", self.data_params["contacts"])


        

        params = {
            'h_diff_weight': 0.6,
            'pos_weight': 0.3,
            'h_shift_weight': 0.2,
            'shift_weight': 0.1,
            'h_contact_weight': 0.8,
            'conatct_weight': 0.4
        }

        stability_label = compute_stability_score(
            pos_diff, ori_diff, shift_pos, shift_ori, contacts,
            self.data_params["position_difference"], self.data_params["orientation_difference"], self.data_params["shift_position"], 
            self.data_params["shift_orientation"], self.data_params["contacts"],
            params=params
        )

        if self.data_params["max_score"] is not None:
            raw_clamped = max(min(stability_label, self.data_params["max_score"]), self.data_params["min_score"])
            stability_label = to_tanh_range(raw_clamped, self.data_params["min_score"], self.data_params["max_score"])
        
        return (feasibility_label, stability_label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, (cls_lbl, reg_lbl) = self.samples[idx]
        cls_lbl = torch.tensor(cls_lbl, dtype=torch.float32)
        reg_lbl = torch.tensor(reg_lbl, dtype=torch.float32)
        return x, (cls_lbl, reg_lbl)

if __name__ == "__main__":

    data_path = "/home/chris/Chris/placement_ws/src/data/processed_data/data.json"
    with open(data_path, "r") as file:
        all_entries = json.load(file)
    random.shuffle(all_entries)

    N = len(all_entries)
    train_size = int(0.8 * N)
    valid_size = int(0.1 * N)
    test_size  = N - train_size - valid_size

    train_entries = all_entries[:train_size]
    valid_entries = all_entries[train_size:train_size + valid_size]
    test_entries  = all_entries[train_size + valid_size:]

    train_dataset = MyStabilityDataset(train_entries)

    import matplotlib.pyplot as plt

    # Assuming train_dataset is already defined as an instance of MyStabilityDataset
    stability_scores = []
    for _, labels in train_dataset:
        # labels is a tuple (feasibility_label, stability_label)
        stability_scores.append(labels[1].item())  # Convert tensor to Python float

    # Convert to a NumPy array for convenience
    stability_scores_np = np.array(stability_scores)

    # Remove outliers using the IQR method
    q1 = np.percentile(stability_scores_np, 25)
    q3 = np.percentile(stability_scores_np, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out scores outside of the bounds
    filtered_scores = stability_scores_np[(stability_scores_np >= lower_bound) &
                                        (stability_scores_np <= upper_bound)]

    # Plot a histogram of the filtered stability scores
    plt.hist(filtered_scores, bins=50, edgecolor='black')
    plt.xlabel("Stability Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Stability Scores (Outliers Removed)")

    # Set x-axis limits to show only the range of interest
    plt.xlim(lower_bound, upper_bound)
    plt.show()
