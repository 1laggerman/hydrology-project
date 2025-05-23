import os
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np


def parse_hyperparameters(folder_name):
    """
    Dynamically extract hyperparameters from the folder name.
    Parameters:
        folder_name (str): Name of the folder containing the run.
    Returns:
        dict: A dictionary of hyperparameters found in the folder name.
    """
    patterns = {
        "hidden_size": r"hidden_size(\d+)",
        "seq_length": r"seq_length(\d+)",
        "batch_size": r"batch_size(\d+)",
        "learning_rate": r"learning_rate(\d+)",
    }
    params = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        if match:
            if key == "learning_rate":
                matched_value = match.group(1)
                params[key] = float(f"0.{matched_value}")*10
            else:
                # Convert other parameters to integers
                params[key] = int(match.group(1))
    return params


def extract_curves_from_folder(folder_path):
    """
    Extract loss curves from a single folder containing a TensorBoard log file.
    Parameters:
        folder_path (str): Path to the folder containing a TensorBoard log file.
    Returns:
        dict: A dictionary containing extracted curves and metadata.
    """
    params = parse_hyperparameters(os.path.basename(folder_path))
    event_acc = EventAccumulator(folder_path)
    event_acc.Reload()

    train_loss, valid_loss, steps = [], [], []

    if 'train/avg_loss' in event_acc.Tags()['scalars']:
        for e in event_acc.Scalars('train/avg_loss'):
            train_loss.append(e.value)
            steps.append(e.step)

    if 'valid/avg_loss' in event_acc.Tags()['scalars']:
        for e in event_acc.Scalars('valid/avg_loss'):
            valid_loss.append(e.value)

    return {
        **params,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'steps': steps
    }


def plot_learning_curves(curves, title="Learning Curves"):
    """
    Plot learning curves for training and validation loss.
    Parameters:
        curves (list[dict]): Extracted curves containing steps, train_loss, and valid_loss.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    for curve in curves:
        params_str = ', '.join([f"{k}={v}" for k, v in curve.items() if k not in ['train_loss', 'valid_loss', 'steps']])
        plt.plot(curve['steps'], curve['train_loss'], label=f"Train ({params_str})", linestyle='-')
        plt.plot(curve['steps'], curve['valid_loss'], label=f"Validation ({params_str})", linestyle='--')

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()


def process_tensorboard_folders(folders):
    """
    Process TensorBoard log files from multiple folders.
    Parameters:
        folders (list[str]): List of folders, each containing a TensorBoard log file.
    """
    curves = []
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"Invalid folder: {folder}")
            continue
        try:
            curve = extract_curves_from_folder(folder)
            curves.append(curve)
        except Exception as e:
            print(f"Error processing {folder}: {e}")

        if curves:
            plot_learning_curves(curves, title="Learning Curves for All Runs")


def main():
    # List of directories containing TensorBoard logs
    log_folders = [
        r"C:\PhD\Python\NH-shared-flow-rain\nhWrap\neuralhydrology\LSTM_shared_RainDis2Dis\runs"
        r"\Check_loss_zscore_norm_ensemble_hidden_size64_batch_size256_learning_rate0001_same_train_val_1401_170103"
    ]
    process_tensorboard_folders(log_folders)


if __name__ == "__main__":
    main()
