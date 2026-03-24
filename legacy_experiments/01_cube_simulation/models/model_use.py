import sys
import os
import json
import torch
import yaml
import re
import numpy as np
import matplotlib.pyplot as plt

# Add the parent folder to sys.path so `learning_models` is found
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from learning_models.dataset import MyStabilityDataset
from learning_models.model_train import StabilityNet  
from learning_models.process_data_helpers import *

def load_run_hyperparams(model_path):
    """
    Load hyperparameters from hyperparams.json located in the same directory as the model.
    """
    run_dir = os.path.dirname(model_path)
    hyperparams_file = os.path.join(run_dir, "hyperparams.json")
    if os.path.exists(hyperparams_file):
        with open(hyperparams_file, "r") as f:
            hyperparams = json.load(f)
        return hyperparams
    return {}

def load_model(model_path, input_dim=21):
    """
    Load a saved model using hyperparameters from hyperparams.json.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparams = load_run_hyperparams(model_path)
    # hidden_dim = hyperparams.get("hidden_dim", 256)
    # Create a new instance of the model with the saved hyperparameters.
    model = StabilityNet(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single_sample(model, sample):
    """
    Given a single sample (a dict with 'inputs' and 'outputs'), create a mini-dataset,
    preprocess the input, and return both the predicted classification label and the
    regression (stability score) from the model.
    """
    device = next(model.parameters()).device
    mini_dataset = MyStabilityDataset([sample])
    x, _ = mini_dataset[0]  # ignore the label here
    x = x.unsqueeze(0).to(device)  # shape [1, input_dim]
    
    with torch.no_grad():
        cls_out, reg_out = model(x)
        # For classification, apply sigmoid and threshold at 0.5
        pred_cls = (torch.sigmoid(cls_out) > 0.5).float().item()
        pred_reg = reg_out.item()
    return pred_cls, pred_reg

def load_all_models(models_root, input_dim):
    """
    Recursively find every 'stability_net.pth' file under models_root,
    load the model (using hyperparams from the run folder), and return a dict mapping paths to models.
    """
    loaded_models = {}
    for root, dirs, files in os.walk(models_root):
        if "stability_net.pth" in files:
            model_path = os.path.join(root, "stability_net.pth")
            try:
                model = load_model(model_path, input_dim)
                loaded_models[model_path] = model
                print(f"Loaded model from: {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
    return loaded_models

def model_comparisons():
    """
    Go through all run folders, load each model, run inference on a test dataset,
    and compute both the RMSE for regression and the classification accuracy for each model.
    Then, plot the RMSE comparisons.
    """
    models_root = "/home/chris/Chris/placement_ws/src/data/models"
    input_dim = 21
    results = {}  # maps seed -> (rmse, classification_accuracy)

    for root, dirs, files in os.walk(models_root):
        if "stability_net.pth" in files:
            model_path = os.path.join(root, "stability_net.pth")
            seed_match = re.search(r'run_seed_(\d+)', root)
            if seed_match:
                seed_value = int(seed_match.group(1))
            else:
                print(f"No seed found in folder: {root}. Skipping.")
                continue
               
            try:
                model = load_model(model_path, input_dim)
                print(f"Loaded model from: {model_path}")

                # Path to your inference JSON file.
                test_file = "/home/chris/Chris/placement_ws/src/data/processed_data/data_splits/test_data.json"
                with open(test_file, "r") as f:
                    data = json.load(f)
                
                pred_cls_list = []
                actual_cls_list = []
                pred_reg_list = []
                actual_reg_list = []
                
                for i, sample in enumerate(data):
                    pred_cls, pred_reg = predict_single_sample(model, sample)
                    # Get actual labels from dataset
                    # __getitem__(0) returns (x, (cls_lbl, reg_lbl))
                    ds = MyStabilityDataset([sample])
                    actual_cls, actual_reg = ds.__getitem__(0)[1]
                    # Convert actual labels from tensors to Python numbers
                    actual_cls = actual_cls.item()
                    actual_reg = actual_reg.item()
                    
                    pred_cls_list.append(pred_cls)
                    actual_cls_list.append(actual_cls)
                    pred_reg_list.append(pred_reg)
                    actual_reg_list.append(actual_reg)

                # Compute regression RMSE
                reg_errors = np.array(pred_reg_list) - np.array(actual_reg_list)
                rmse = np.sqrt(np.mean(reg_errors ** 2))
                
                # Compute classification accuracy
                pred_cls_arr = np.array(pred_cls_list)
                actual_cls_arr = np.array(actual_cls_list)
                cls_accuracy = np.mean(pred_cls_arr == actual_cls_arr)
                
                print(f"\nFor seed {seed_value}: RMSE = {rmse:.4f}, Classification Accuracy = {cls_accuracy*100:.2f}%")
                results[seed_value] = (rmse, cls_accuracy)

            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")

    if results:
        # Identify the best model based on RMSE
        best_seed = min(results, key=lambda s: results[s][0])
        best_rmse, best_acc = results[best_seed]
        print(f"Best RMSE: {best_rmse:.4f} from seed {best_seed} (Accuracy: {best_acc*100:.2f}%)")

        # Plot RMSE comparisons
        seeds = sorted(results.keys())
        rmses = [results[s][0] for s in seeds]

        plt.figure(figsize=(10, 6))
        plt.scatter(seeds, rmses, color='skyblue', s=100)
        plt.xlabel("Seed")
        plt.ylabel("RMSE")
        plt.title("Model RMSE Comparisons (Zoomed)")
        rmse_min, rmse_max = min(rmses), max(rmses)
        margin = (rmse_max - rmse_min) * 0.1 if rmse_max != rmse_min else 0.01
        plt.ylim(rmse_min - margin, rmse_max + margin)
        plt.show()

        # Plot classification accuracy comparisons
        accuracies = [results[s][1]*100 for s in seeds]  # convert to percentage

        plt.figure(figsize=(10, 6))
        plt.bar(seeds, accuracies, color='lightgreen')
        plt.xlabel("Seed")
        plt.ylabel("Accuracy (%)")
        plt.title("Model Classification Accuracy Comparisons")
        plt.ylim(0, 100)
        plt.show()

    else:
        print("No model results to plot.")

def single_model(model_seed):
    """
    Find and load the model with the given seed.
    """
    input_dim = 21
    models_root = "/home/chris/Chris/placement_ws/src/data/models"
    target_model_path = None

    for root, dirs, files in os.walk(models_root):
        if "stability_net.pth" in files:
            model_path = os.path.join(root, "stability_net.pth")
            seed_match = re.search(r'run_seed_(\d+)', root)
            if seed_match:
                seed_value = int(seed_match.group(1))
                if seed_value == model_seed:
                    target_model_path = model_path
                    break

    if target_model_path is None:
        print(f"Model with seed {model_seed} not found.")
        return None

    try:
        model = load_model(target_model_path, input_dim)
        print(f"Loaded model from: {target_model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {target_model_path}: {e}")
        return None


def model_test(seed):
    # Test a single model with a specific seed (change the seed value as needed):
    model = single_model(seed)
    
    test_file = "/home/chris/Chris/placement_ws/src/data/processed_data/data_splits/test_data.json"

    with open(test_file, "r") as f:
        data = json.load(f)
    
    pred_cls_list = []
    actual_cls_list = []
    pred_reg_list = []
    actual_reg_list = []
    
    for i, sample in enumerate(data):
        pred_cls, pred_reg = predict_single_sample(model, sample)
        # Using the dataset helper to get actual labels:
        ds = MyStabilityDataset([sample])
        actual_cls, actual_reg = ds.__getitem__(0)[1]
        actual_cls = actual_cls.item()
        actual_reg = actual_reg.item()

        pred_cls_list.append(pred_cls)
        actual_cls_list.append(actual_cls)
        pred_reg_list.append(pred_reg)
        actual_reg_list.append(actual_reg)
        
        print(f"Sample {i} => Predicted: (Cls: {pred_cls:.0f}, Reg: {pred_reg:.4f}) | Actual: (Cls: {actual_cls:.0f}, Reg: {actual_reg:.4f})")

    # Calculate regression RMSE
    reg_errors = np.array(pred_reg_list) - np.array(actual_reg_list)
    rmse = np.sqrt(np.mean(reg_errors ** 2))
    print(f"\nRegression RMSE: {rmse:.4f}")
    
    # Calculate classification accuracy
    pred_cls_arr = np.array(pred_cls_list)
    actual_cls_arr = np.array(actual_cls_list)
    cls_accuracy = np.mean(pred_cls_arr == actual_cls_arr)
    print(f"Classification Accuracy: {cls_accuracy*100:.2f}%")
    
    # --- Plotting the Comparisons ---
    # Scatter plot for regression: Actual vs Predicted
    num_samples = len(pred_reg_list)
    sample_indices = np.random.choice(num_samples, min(200, num_samples), replace=False)
    sampled_preds = [pred_reg_list[i] for i in sample_indices]
    sampled_actuals = [actual_reg_list[i] for i in sample_indices]

    plt.figure(figsize=(8, 6))
    plt.scatter(sampled_actuals, sampled_preds, alpha=0.6)
    plt.plot([min(sampled_actuals), max(sampled_actuals)],
             [min(sampled_actuals), max(sampled_actuals)], 'r--', label="Ideal")
    plt.xlabel("Actual Stability Score")
    plt.ylabel("Predicted Stability Score")
    plt.title("Sampled Predicted vs Actual Regression Scores")
    plt.legend()
    plt.show()

    # Histogram of prediction errors (regression)
    plt.figure(figsize=(8, 6))
    plt.hist(reg_errors, bins=50, alpha=0.7, color='green')
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Regression Prediction Errors")
    plt.show()





if __name__ == "__main__":
    # Uncomment one of the following to run comparisons or test a single model.
    # model_comparisons()
    seed = 30395149
    model_test(seed)
    
    
