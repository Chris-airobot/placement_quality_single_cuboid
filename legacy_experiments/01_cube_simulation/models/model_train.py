import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb  # <--- Import W&B

# Add the parent folder to sys.path so `learning_models` is found
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from learning_models.dataset import MyStabilityDataset

# Base directory for saving models and images
DIR = "/home/chris/Chris/placement_ws/src/data/models/final_test"

###############################################################################
# 1) MODEL DEFINITION (Two-head network: classification and regression)
###############################################################################
class StabilityNet(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 1)  # Binary classification (logits)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # yields [-1, 1]
        )
     # Regression (stability score)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        cls_out = self.class_head(x)
        reg_out = self.reg_head(x)
        return cls_out, reg_out

###############################################################################
# 2) LOSS FUNCTIONS
###############################################################################
def combined_loss(pred_cls, label_cls, pred_reg, label_reg):
    # print(f"At this time, pred_cls: {pred_cls}, label_cls: {label_cls}, pred_reg: {pred_reg}, label_reg: {label_reg}")
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.MSELoss()
    
    # Flatten predictions and labels to 1D
    pred_cls = pred_cls.view(-1)
    label_cls = label_cls.view(-1)
    loss_cls = criterion_cls(pred_cls, label_cls)
    
    pred_reg = pred_reg.view(-1)
    label_reg = label_reg.view(-1)
    loss_reg = criterion_reg(pred_reg, label_reg)
    # print(f"and your loss is: {loss_cls + loss_reg}, loss_cls is: {loss_cls}, loss_reg is: {loss_reg}")
    return loss_cls + loss_reg, loss_cls, loss_reg

###############################################################################
# 3) HELPER FUNCTIONS FOR TRAINING AND EVALUATION
###############################################################################
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    for x, (cls_lbl, reg_lbl) in loader:  
        x = x.float().to(device)
        cls_lbl = cls_lbl.float().to(device)
        reg_lbl = reg_lbl.float().to(device)

        optimizer.zero_grad()
        pred_cls, pred_reg = model(x)
        loss, loss_cls, loss_reg = combined_loss(pred_cls, cls_lbl, pred_reg, reg_lbl)
        # print(f"for one data, loss is: {loss_reg}")
        # print(f"And corresponding labels are: {pred_reg}, the actual labels are: {reg_lbl}")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_reg_loss += loss_reg.item()

    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)

    # print(f"Overall for one epoch, loss is: {avg_loss}, loss_cls is: {avg_cls_loss}, loss_reg is: {avg_reg_loss}")
    return avg_loss, avg_cls_loss, avg_reg_loss

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    with torch.no_grad():
        for x, (cls_lbl, reg_lbl) in loader:
            x = x.float().to(device)
            cls_lbl = cls_lbl.float().to(device)
            reg_lbl = reg_lbl.float().to(device)
            pred_cls, pred_reg = model(x)
            loss, loss_cls, loss_reg = combined_loss(pred_cls, cls_lbl, pred_reg, reg_lbl)
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()

    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_reg_loss = total_reg_loss / len(loader)
    return avg_loss, avg_cls_loss, avg_reg_loss

def save_model(model, path='stability_net.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def sample_hyperparams():
    # hidden_dim = 2 ** random.randint(6, 9)  # e.g., 64 to 512
    batch_size = random.choice([16, 32, 64, 128])
    log_lr = random.uniform(-5, -3)
    learning_rate = 10 ** log_lr
    optimizer = random.choice(['adam', 'sgd'])
    seed = random.randint(0, 99999999)
    scheduler_factor = random.choice([0.5, 0.6, 0.7])
    scheduler_patience = random.randint(2, 6)
    num_epochs = random.choice([200,300])
    params = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'optimizer': optimizer,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'seed': seed,
        # 'hidden_dim': hidden_dim
    }

    # params = {
    #     'batch_size': 16,
    #     'num_epochs': 300,
    #     'learning_rate': 1.1897909925768335e-05,
    #     'optimizer': 'adam',
    #     'scheduler_factor': 0.5,
    #     'scheduler_patience': 3,
    #     'seed': seed,
    #     # 'hidden_dim': hidden_dim
    # }
    return params

def plot_and_save_loss(train_losses, valid_losses, 
                       title, run_fig_path,  train_label="Train Loss", valid_label="Validation Loss", 
                       image_file_path=None):
    """
    Plots and saves the training and validation losses.
    
    Parameters:
    - train_losses: List of training losses per epoch.
    - valid_losses: List of validation losses per epoch.
    - title: Title of the plot.
    - y_label: Label for the y-axis.
    - run_fig_path: File path to save the figure in the current run directory.
    - image_file_path: File path to save the figure in the central images folder.
    - train_label: Legend label for training loss.
    - valid_label: Legend label for validation loss.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label=train_label)
    plt.plot(valid_losses, label=valid_label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save to the run directory
    plt.savefig(run_fig_path)
    print(f"{title} figure saved to {run_fig_path}")
    
    # Save to the central images folder
    if image_file_path is not None:
        plt.savefig(image_file_path)
        print(f"Plot saved to {image_file_path}")
    
    # Close the plot to avoid overlapping figures in subsequent calls
    plt.close()

###############################################################################
# 4) MAIN TRAINING FUNCTION WITH W&B LOGGING AND SAVING HYPERPARAMETERS & PLOTS
###############################################################################
def main(params, data_folder):
    # Initialize W&B run
    run = wandb.init(project="stability-regression",
                     config=params,
                     name=f"run_seed_{params['seed']}")
    run_dir = os.path.join(DIR, run.name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save hyperparameters to JSON in run_dir
    hyperparams_path = os.path.join(run_dir, "hyperparams.json")
    with open(hyperparams_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Hyperparameters saved to {hyperparams_path}")

    # Set random seed
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, seed={seed}")

    # Load data
    train_path = data_folder + "/train_data.json"
    valid_path = data_folder + "/valid_data.json"
    test_path = data_folder + "/test_data.json"

    with open(train_path, "r") as file:
        train_entries = json.load(file)
    with open(valid_path, "r") as file:
        valid_entries = json.load(file)
    with open(test_path, "r") as file:
        test_entries = json.load(file)
    # print(f"Train: {len(train_entries)}, Valid: {len(valid_entries)}, Test: {len(test_entries)}")

    random.shuffle(train_entries)

    
    train_dataset = MyStabilityDataset(train_entries)
    valid_dataset = MyStabilityDataset(valid_entries)
    test_dataset  = MyStabilityDataset(test_entries)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=params['batch_size'], shuffle=False)

    # Build model & optimizer
    model = StabilityNet(input_dim=21).to(device)
    if params['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=params['scheduler_factor'], 
        patience=params['scheduler_patience'], 
        verbose=True
    )

    # Training loop
    train_losses, valid_losses = [], []
    train_cls_losses, valid_cls_losses = [], []
    train_reg_losses, valid_reg_losses = [], []

    for epoch in range(params['num_epochs']):
        train_loss, train_cls_loss, train_reg_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_cls_loss, val_reg_loss = eval_model(model, valid_loader, device)

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        train_cls_losses.append(train_cls_loss)
        valid_cls_losses.append(val_cls_loss)
        
        train_reg_losses.append(train_reg_loss)
        valid_reg_losses.append(val_reg_loss)

        print(f"Epoch {epoch+1}/{params['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Cls: {val_cls_loss:.4f}, Reg: {val_reg_loss:.4f})")

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_cls_loss": train_cls_loss,
            "train_reg_loss": train_reg_loss,
            "val_loss": val_loss,
            "val_cls_loss": val_cls_loss,
            "val_reg_loss": val_reg_loss,
        })

        scheduler.step(val_loss)

    # Final evaluation & Save
    test_loss, test_cls_loss, test_reg_loss = eval_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} (Cls: {test_cls_loss:.4f}, Reg: {test_reg_loss:.4f})")
    wandb.log({"test_loss": test_loss, "test_cls_loss": test_cls_loss, "test_reg_loss": test_reg_loss})

    model_path = os.path.join(run_dir, "stability_net.pth")
    save_model(model, model_path)

    images_dir = os.path.join(DIR, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Overall Loss Plot
    overall_run_fig = os.path.join(run_dir, "train_val_loss.png")
    overall_image_fig = os.path.join(images_dir, f"seed_{params['seed']}.png")
    plot_and_save_loss(train_losses, valid_losses, "Train vs Val Loss", overall_run_fig, image_file_path=overall_image_fig)

    # Classification Loss Plot
    cls_run_fig = os.path.join(run_dir, "cls_loss.png")
    plot_and_save_loss(train_cls_losses, valid_cls_losses, "Train vs Val Classification Loss", cls_run_fig, "Train Classification Loss", "Validation Classification Loss")

    # Regression Loss Plot
    reg_run_fig = os.path.join(run_dir, "reg_loss.png")
    plot_and_save_loss(train_reg_losses, valid_reg_losses, "Train vs Val Regression Loss", reg_run_fig, "Train Regression Loss", "Validation Regression Loss")

    wandb.finish()

###############################################################################
# 5) ENTRY POINT
###############################################################################
if __name__ == "__main__":
    total_experiment = 10
    for i in range(total_experiment):
        params = sample_hyperparams()
        print(f"\n=== Experiment {i+1}/{total_experiment} with {params} ===\n")
        main(params, "/home/chris/Chris/placement_ws/src/data/processed_data/data_splits")
